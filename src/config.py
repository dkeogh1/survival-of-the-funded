"""Central configuration: KFS column names, wave structure, and industry mapping.

See docs/VARIABLE_MAPPING.md for the full thesis-variable -> KFS-column crosswalk.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DTA = ROOT / "data" / "raw" / "KFS8_PublicUse.dta"
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "output"

WAVES = list(range(8))          # 0 = baseline (2004) ... 7 = 7th follow-up (2011)
FOLLOWUP_WAVES = list(range(1, 8))
PRIMARY = "owner_01"            # primary founder index

# ---------------------------------------------------------------- outcomes
STATUS = [f"final_status_code_{w}" for w in WAVES]
OUT_OF_BUSINESS = [f"a10_out_of_business_{w}" for w in FOLLOWUP_WAVES]
REVENUE = [f"tot_revenue_r_{w}" for w in WAVES]
EMPLOYMENT = [f"c5_num_employees_{w}" for w in WAVES]

# Disposition codes (KFS codebook ch.2, Appendix B)
ALIVE_CODES = {10, 30}                       # telephone/CATI or web complete
PERMANENT_STOP_CODE = 463                     # No Longer in Business
TEMP_STOP_CODE = 431                          # Temporarily Stopped Operations
DROP_FIRM_CODES = {465, 468}                  # pre-2004 start / duplicate -> ineligible
# a10_out_of_business reason codes
A10_SOLD, A10_MERGED, A10_OOB_A, A10_OOB_B, A10_TEMP = 1, 2, 3, 4, 5

# --------------------------------------------------- human & social capital
HC = {
    "foundered":          f"g9_education_{PRIMARY}",      # education level 1-10
    "founderworkexp":     f"g2_work_exp_{PRIMARY}",       # years professional exp
    "founderexp":         f"g3a_oth_bus_{PRIMARY}",       # # businesses founded (dropped)
    "founderexpsameind":  f"g3b_bus_same_ind_{PRIMARY}",  # founded in relevant industry
}

# ---------------------------------------------------------------- financing
# "ever received over observed waves"; each maps to one or more KFS column stems
FINANCING = {
    "eqangels":    ["f3c_eq_invest_angels"],
    "eqcompanies": ["f3d_eq_invest_companies"],
    "eqvc":        ["f3f_eq_invest_vent_cap"],
    "eqgovt":      ["f3e_eq_invest_govt"],
    "eqfff":       ["f3a_eq_invest_spouse", "f3b_eq_invest_parents",
                    "f11a_bus_loans_fam", "f7a_pers_loan_fam", "f9a_pers_loan_fam"],
    "debtfin":     ["f11a_bus_loans_bank", "f11a_bus_loans_nonbank",
                    "f11a_bus_loans_govt", "f11a_bus_loans_owner",
                    "f11a_bus_loans_emp", "f11a_bus_loans_other_bus",
                    "f11a_bus_loans_other_ind",
                    "f7a_pers_loan_bank", "f7a_pers_loan_other",
                    "f9a_pers_loan_bank", "f9a_pers_loan_other"],
}

# ------------------------------------------------------- competitive advantage
# subtype channels only from wave 3 (2007); back-filled to earlier waves
COMPADV = {
    "univcompadv":   "d2a_compadv_univ_reason",
    "compcompadv":   "d2a_compadv_comp_reason",
    "patentcompadv": "d2a_compadv_patents_reason",
    "govlabcompadv": "d2a_compadv_govlab_reason",
}
COMPADV_OVERALL = "d2_comp_advantage"   # binary, all waves

# ------------------------------------------------------------------------ IP
IP = {
    "totcr":      "total_copyrights_0",
    "tottm":      "total_trademarks_0",
    "totpatents": "total_patents_0",
}

# ----------------------------------------------------------- demographics (baseline)
DEMOG = {
    "foundhisp":  f"g5_hisp_origin_{PRIMARY}_0",
    "foundamind": f"g6_race_amind_{PRIMARY}_0",
    "foundasian": f"g6_race_asian_{PRIMARY}_0",
    "foundblack": f"g6_race_black_{PRIMARY}_0",
    "foundwhite": f"g6_race_white_{PRIMARY}_0",
    "foundmale":  f"g10_gender_{PRIMARY}_0",
    "foundage":   f"age_{PRIMARY}_r_0",
}

# ---------------------------------------------------------------- industry
NAICS_COL = "naics_code_0"
HIGHTECH_COL = "hightech_0"
# NAICS 2-digit prefix -> thesis dummy. Unlisted sectors fall into the reference group.
NAICS_TO_INDUSTRY = {
    "21": "mining", "22": "ut", "23": "con",
    "31": "manu", "32": "manu", "33": "manu",
    "48": "tnw", "49": "tnw",
    "51": "inf", "52": "finser", "53": "re", "54": "profser",
    "55": "management", "56": "wm", "61": "eduser", "71": "rec", "72": "food",
}
INDUSTRY_DUMMIES = ["mining", "ut", "con", "manu", "tnw", "inf", "finser", "re",
                    "profser", "management", "wm", "eduser", "rec", "food"]

# ---------------------------------------------------------------- interactions
INTERACTIONS = {
    "ednet":     ("foundered", "univcompadv"),
    "indnet":    ("founderexpsameind", "compcompadv"),
    "netfwex_2": ("founderworkexp", "compcompadv"),
    "netang_si": ("founderexpsameind", "eqangels"),
    "netvc_si":  ("founderexpsameind", "eqvc"),
    "netcomp_si":("founderexpsameind", "eqcompanies"),
}
