import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import re
import gc

st.set_page_config(page_title="SA Race Coast Wizard", page_icon="🏇", layout="centered")

st.title("🏇 SA Race Coast / Computerform Wizard")
st.write("Upload any Race Coast / Computerform PDF to generate Value Picks, Dark Horses, and Exotic Bet Builders.")

# ---------------------------------------------------------
# 1. FAST, LOW-MEMORY COMPUTERFORM PDF PARSER
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def parse_sa_pdf(file_bytes):
    runners = []
    meeting_name = "South African Race Meeting"
    current_race = 1
    current_dist = 1200

    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text("text")
            lines = [line.strip() for line in text.split("\n") if line.strip()]

            # Detect meeting venue on early pages
            if page_idx < 3:
                for venue in ["KENILWORTH", "GREYVILLE", "SCOTTSVILLE", "TURFFONTEIN", "VAAL", "FAIRVIEW", "DURBANVILLE"]:
                    if venue in text.upper():
                        meeting_name = f"Hollywoodbets {venue.title()}" if venue in ["KENILWORTH", "GREYVILLE", "SCOTTSVILLE", "DURBANVILLE"] else venue.title()
                        break

            # Detect Race Number & Distance (e.g., "Race 1", "1200m")
            race_match = re.search(r"\bRace\s+(\d{1,2})\b", text, re.IGNORECASE)
            if race_match:
                current_race = int(race_match.group(1))

            dist_match = re.search(r"\b(\d{4})\s*m\b", text, re.IGNORECASE)
            if dist_match:
                current_dist = int(dist_match.group(1))

            # Parse runner blocks (Number, Horse Name, Draw, Weight, Merit Rating, Form)
            for i, line in enumerate(lines):
                # Look for typical Computerform runner header: e.g. "1  HORSE NAME  (4)  60.0" or "1 HORSE NAME"
                m = re.match(r"^(\d{1,2})\s+([A-Z][A-Z\s\'\-]{2,28})(?:\s+\((\d{1,2})\))?", line)
                if m:
                    saddle_no = int(m.group(1))
                    horse_name = m.group(2).strip()
                    
                    # Filter out non-horse headers
                    skip_words = ["RACE", "WIN", "PLACE", "SWINGER", "EXACTA", "TRIFECTA", "QUARTET", "JACKPOT", "PICK", "BIPOT", "TOTAL", "TIME"]
                    if any(w in horse_name.split() for w in skip_words) or saddle_no > 24:
                        continue

                    block_text = " ".join(lines[i:min(i + 8, len(lines))])

                    # Extract Draw (e.g. "(5)" or "Dr 5")
                    draw = int(m.group(3)) if m.group(3) else 0
                    if draw == 0:
                        d_match = re.search(r"\((\d{1,2})\)", block_text)
                        if d_match:
                            draw = int(d_match.group(1))

                    # Extract Weight (e.g. 58.5 or 60.0)
                    w_match = re.search(r"\b(5[0-9]\.\d|6[0-2]\.\d)\b", block_text)
                    weight = float(w_match.group(1)) if w_match else 58.0

                    # Extract Merit Rating (MR)
                    mr_match = re.search(r"\bMR[:\s]*(\d{2,3})\b", block_text)
                    mr = int(mr_match.group(1)) if mr_match else 75

                    # Extract recent finishing positions (e.g. "4P12 2K14 1G12")
                    form_runs = re.findall(r"\b(\d)[A-Z]\d{2}\b", block_text)
                    if not form_runs:
                        form_runs = re.findall(r"\b(\d)\b", block_text)[:4]
                    finishes = [int(x) if int(x) > 0 else 10 for x in form_runs[:4]]
                    avg_finish = np.mean(finishes) if finishes else 6.0

                    # Detect Gear Upgrades (Blinkers, Cheekpieces, Tongue Tie, Alumites)
                    gear_upgrade = 1.0 if re.search(r"\b(Blinkers|B\+|C\/P|Cheek|Tongue|T\/T|Pacifiers)\b", block_text, re.IGNORECASE) else 0.0

                    runners.append({
                        "Race": current_race,
                        "Distance": current_dist,
                        "No": saddle_no,
                        "Horse": horse_name,
                        "Draw": draw if draw > 0 else saddle_no,
                        "Weight": weight,
                        "MR": mr,
                        "AvgFinish": avg_finish,
                        "GearUpgrade": gear_upgrade
                    })

            page = None  # Free page memory immediately

    gc.collect()

    if not runners:
        return pd.DataFrame(), meeting_name

    df = pd.DataFrame(runners).drop_duplicates(subset=["Race", "No", "Horse"]).reset_index(drop=True)
    return df, meeting_name


# ---------------------------------------------------------
# 2. WIZARD SCORING ENGINE
# ---------------------------------------------------------
def run_wizard_engine(df, weights):
    scored_list = []

    for race_num, group in df.groupby("Race"):
        g = group.copy()

        # Normalize Speed & Weight (lighter weight + better MR relative to weight = higher score)
        weight_adv = (g["Weight"].max() - g["Weight"]) / max(1.0, (g["Weight"].max() - g["Weight"].min()))
        mr_norm = (g["MR"] - g["MR"].min()) / max(1.0, (g["MR"].max() - g["MR"].min()))
        g["speed_score"] = (0.6 * mr_norm) + (0.4 * weight_adv)

        # Normalize Class & MR Drop
        g["class_score"] = mr_norm

        # Normalize Form & Fitness (lower average finish position = better)
        g["form_score"] = (g["AvgFinish"].max() - g["AvgFinish"]) / max(1.0, (g["AvgFinish"].max() - g["AvgFinish"].min()))

        # Normalize Draw Bias (lower draw preferred around turns / standard bias)
        g["draw_score"] = (g["Draw"].max() - g["Draw"]) / max(1.0, (g["Draw"].max() - g["Draw"].min()))

        # Gear Upgrades
        g["gear_score"] = g["GearUpgrade"]

        # Weighted Total Score (0 to 100 scale)
        total_w = sum(weights.values()) or 1.0
        g["WizardScore"] = (
            (g["speed_score"] * weights["speed"]) +
            (g["class_score"] * weights["class"]) +
            (g["form_score"] * weights["form"]) +
            (g["draw_score"] * weights["draw"]) +
            (g["gear_score"] * weights["gear"])
        ) / total_w * 100

        g["WizardScore"] = g["WizardScore"].round(1)
        g = g.sort_values(by="WizardScore", ascending=False).reset_index(drop=True)
        g["Rank"] = g.index + 1
        scored_list.append(g)

    return pd.concat(scored_list, ignore_index=True)


# ---------------------------------------------------------
# 3. STREAMLIT UI & EXPLICIT TRIGGER BUTTON
# ---------------------------------------------------------
uploaded_pdf = st.file_uploader(
    "Upload South African Race Coast / Computerform PDF",
    type=["pdf"]
)

w_speed = st.slider("Speed & Weight", 0.0, 1.0, 0.35, 0.05)
w_class = st.slider("Class & MR Drop", 0.0, 1.0, 0.25, 0.05)
w_form = st.slider("Form & Fitness", 0.0, 1.0, 0.20, 0.05)
w_draw = st.slider("Draw Bias", 0.0, 1.0, 0.10, 0.05)
w_gear = st.slider("Gear Upgrades", 0.0, 1.0, 0.10, 0.05)

# Explicit trigger button below Gear Upgrades slider
if st.button("Generate Picks", type="primary", use_container_width=True):
    if uploaded_pdf is None:
        st.warning("Please upload an official Race Coast Computerform PDF first.")
    else:
        with st.spinner("Parsing Computerform PDF & running Wizard Engine..."):
            pdf_bytes = uploaded_pdf.getvalue()
            raw_df, meeting_name = parse_sa_pdf(pdf_bytes)

        if raw_df.empty:
            st.error("Could not find racecard runner blocks in this PDF. Please upload an official Race Coast Computerform PDF.")
        else:
            weights = {
                "speed": w_speed,
                "class": w_class,
                "form": w_form,
                "draw": w_draw,
                "gear": w_gear
            }
            scored_df = run_wizard_engine(raw_df, weights)

            st.success(f"Loaded **{meeting_name}** ({scored_df['Race'].nunique()} Races Parsed)")

            # Display Race-by-Race Picks & Exotic Bet Builders
            for race_num, r_df in scored_df.groupby("Race"):
                with st.expander(f"Race {race_num} ({r_df['Distance'].iloc[0]}m) — Top Selection: #{r_df.iloc[0]['No']} {r_df.iloc[0]['Horse']}", expanded=True):
                    top_pick = r_df.iloc[0]
                    value_pick = r_df.iloc[1] if len(r_df) > 1 else top_pick
                    dark_horse = r_df[r_df["GearUpgrade"] > 0].iloc[0] if not r_df[r_df["GearUpgrade"] > 0].empty else (r_df.iloc[2] if len(r_df) > 2 else top_pick)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("🏆 Top Pick", f"#{top_pick['No']} {top_pick['Horse']}", f"Score: {top_pick['WizardScore']}")
                    col2.metric("💎 Value Pick", f"#{value_pick['No']} {value_pick['Horse']}", f"Score: {value_pick['WizardScore']}")
                    col3.metric("🐎 Dark Horse", f"#{dark_horse['No']} {dark_horse['Horse']}", f"Draw: {dark_horse['Draw']}")

                    # Exotic Bet Builder suggestions
                    top4_nos = [str(x) for x in r_df.head(4)["No"].tolist()]
                    if len(top4_nos) >= 3:
                        st.info(
                            f"**Exotic Bet Builder (Race {race_num}):**\n"
                            f"- **Swinger / Exacta Box:** {', '.join(top4_nos[:3])}\n"
                            f"- **Trifecta / Quartet Box:** {', '.join(top4_nos)}"
                        )

                    st.dataframe(
                        r_df[["Rank", "No", "Horse", "Draw", "Weight", "MR", "WizardScore"]],
                        use_container_width=True,
                        hide_index=True
                    )
