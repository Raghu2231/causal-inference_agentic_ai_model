from dataclasses import dataclass
from typing import Dict, Tuple, List

@dataclass
class HopperOutput():
                   
    skyrizi: str = 'z_abv_cws_mabi_omni_chn_anl_hcp.suggestions_measurement__skyrizi_master_input'
    vraylar: str = 'z_abv_cws_mabi_omni_chn_anl_hcp.suggestions_measurement__vraylar_master_input'
    simdata: str = 'z_abv_cws_mabi_omni_chn_anl_hcp.suggestions_measurement__simdata_master_input'

VRAYLAR_SNOWFLAKE_POINTER: str = 'Vraylar_suggestion_veeva_npp_01242025'

columns_to_exclude: Dict[Tuple[str], List[str]] = {
    ('npp', 'ho_email'): [
        'covariate_path_b_hcp_total_ho_emails_last_8_weeks',
    ],
    # Suggestion_type = all includes email, call, and insight so all need to exclude irep_emails
    ('email', 'email'): [
        'covariate_path_b_hcp_total_irep_emails_last_8_weeks',
    ],
    ('call', 'email'): [
        'covariate_path_b_hcp_total_irep_emails_last_8_weeks',
    ],
    ('insight', 'email'): [
        'covariate_path_b_hcp_total_irep_emails_last_8_weeks',
    ],
    ('all', 'email'): [
        'covariate_path_b_hcp_total_irep_emails_last_8_weeks',
    ],
}