"""NOTA AI: instrument-specific pedagogical guidance and safe analysis scope."""
from typing import Optional

INSTRUMENTS = {
    'dutor': {'id': 'dutor', 'name_uz': 'Dutor', 'group': 'milliy', 'family': 'plucked', 'analysis_scope': 'Bir ovozli kuy, zarb va tor almashinuvi', 'teacher_focus': 'Tor almashinuvi, mizrob ritmi, rezonans', 'validated_model': False, 'professional_grading_available': False},
    'rubob': {'id': 'rubob', 'name_uz': 'Rubob', 'group': 'milliy', 'family': 'plucked', 'analysis_scope': 'Bir ovozli kuy va zarbli ijro', 'teacher_focus': 'Mizrob, tremolo, tovush aniqligi', 'validated_model': False, 'professional_grading_available': False},
    'tanbur': {'id': 'tanbur', 'name_uz': 'Tanbur', 'group': 'milliy', 'family': 'plucked', 'analysis_scope': 'Bir ovozli kuy va cho‘zilgan tovushlar', 'teacher_focus': 'Mizrob, nolish, tovush davomiyligi', 'validated_model': False, 'professional_grading_available': False},
    'gijjak': {'id': 'gijjak', 'name_uz': 'G‘ijjak', 'group': 'milliy', 'family': 'bowed', 'analysis_scope': 'Bir ovozli kuy', 'teacher_focus': 'Kamon almashinuvi, intonatsiya, legato', 'validated_model': False, 'professional_grading_available': False},
    'chang': {'id': 'chang', 'name_uz': 'Chang', 'group': 'milliy', 'family': 'struck', 'analysis_scope': 'Ko‘p ovozli va tez zarbli ijro', 'teacher_focus': 'Zarb aniqligi, rezonans, ovozlar ajralishi', 'validated_model': False, 'professional_grading_available': False},
    'nay': {'id': 'nay', 'name_uz': 'Nay', 'group': 'milliy', 'family': 'wind', 'analysis_scope': 'Bir ovozli kuy', 'teacher_focus': 'Nafas, fraza, intonatsiya', 'validated_model': False, 'professional_grading_available': False},
    'doira': {'id': 'doira', 'name_uz': 'Doira', 'group': 'milliy', 'family': 'percussion', 'analysis_scope': 'Ritmik usul; nota balandligi emas', 'teacher_focus': 'Usul, zarb va pauza aniqligi', 'validated_model': False, 'professional_grading_available': False},
    'piano': {'id': 'piano', 'name_uz': 'Fortepiano', 'group': 'klassik', 'family': 'keyboard', 'analysis_scope': 'Ko‘p ovozli, akkordli ijro', 'teacher_focus': 'Ikki qo‘l uyg‘unligi, pedal, dinamika', 'validated_model': False, 'professional_grading_available': False},
    'violin': {'id': 'violin', 'name_uz': 'Skripka', 'group': 'klassik', 'family': 'bowed', 'analysis_scope': 'Bir ovozli kuy', 'teacher_focus': 'Intonatsiya, smichok, artikulyatsiya', 'validated_model': False, 'professional_grading_available': False},
    'cello': {'id': 'cello', 'name_uz': 'Violonchel', 'group': 'klassik', 'family': 'bowed', 'analysis_scope': 'Bir ovozli kuy', 'teacher_focus': 'Intonatsiya, smichok, legato', 'validated_model': False, 'professional_grading_available': False},
    'guitar': {'id': 'guitar', 'name_uz': 'Gitara', 'group': 'klassik', 'family': 'plucked', 'analysis_scope': 'Akkordli va ko‘p ovozli ijro', 'teacher_focus': 'Barmoqlar, akkordlar, artikulyatsiya', 'validated_model': False, 'professional_grading_available': False},
    'flute': {'id': 'flute', 'name_uz': 'Fleyta', 'group': 'klassik', 'family': 'wind', 'analysis_scope': 'Bir ovozli kuy', 'teacher_focus': 'Nafas, artikulyatsiya, fraza', 'validated_model': False, 'professional_grading_available': False},
    'saxophone': {'id': 'saxophone', 'name_uz': 'Saksofon', 'group': 'boshqa', 'family': 'wind', 'analysis_scope': 'Bir ovozli kuy', 'teacher_focus': 'Nafas, intonatsiya, artikulyatsiya', 'validated_model': False, 'professional_grading_available': False},
    'clarinet': {'id': 'clarinet', 'name_uz': 'Klarnet', 'group': 'boshqa', 'family': 'wind', 'analysis_scope': 'Bir ovozli kuy', 'teacher_focus': 'Nafas, registr o‘tishlari, intonatsiya', 'validated_model': False, 'professional_grading_available': False},
    'accordion': {'id': 'accordion', 'name_uz': 'Akkordeon', 'group': 'boshqa', 'family': 'keyboard', 'analysis_scope': 'Ko‘p ovozli, akkordli ijro', 'teacher_focus': 'Mex harakati, qo‘llar uyg‘unligi, dinamika', 'validated_model': False, 'professional_grading_available': False},
}

def get_instrument(key: Optional[str]):
    if key is None:
        return None
    return INSTRUMENTS.get(key)

def instrument_guidance(profile):
    if profile is None:
        return {"status":"instrument_not_selected", "message":"Cholg‘u tanlanmagan; umumiy tahlil qo‘llanadi."}
    family=profile["family"]
    unsupported=family in ("keyboard", "percussion", "struck") or profile["id"] == "guitar"
    return {
        "instrument": profile["name_uz"],
        "analysis_compatibility": "limited" if unsupported else "experimental_monophonic",
        "pitch_assessment_supported": family != "percussion",
        "pedagogical_focus": profile["teacher_focus"],
        "recommended_practice": "Sekin tempda kichik frazalarni mashq qiling; xato joylarni takrorlang.",
        "caution": ("Ko‘p ovozli yoki zarbli ijro uchun bir ovozli tovush ajratish ishonchli emas." if unsupported else "Cholg‘u uchun alohida tasdiqlangan AI modeli mavjud emas; tahlil taxminiy."),
        "professional_grading_available": False,
    }
