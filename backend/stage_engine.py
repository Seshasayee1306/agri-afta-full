from datetime import datetime

def calculate_days_after_sowing(sowing_date, current_date):
    sowing = datetime.strptime(sowing_date, "%Y-%m-%d")
    current = datetime.strptime(current_date, "%Y-%m-%d")
    return (current - sowing).days


def identify_growth_stage(days, total_duration=120):
    stage_length = total_duration / 4

    if days <= stage_length:
        return "germination"
    elif days <= stage_length * 2:
        return "vegetative"
    elif days <= stage_length * 3:
        return "flowering"
    else:
        return "harvest"