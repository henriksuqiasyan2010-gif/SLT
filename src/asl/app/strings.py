"""Every interface text of the app in Armenian, Russian and English.

    strings.set_language("ru")
    strings.tr("btn_record")                       # -> 'Записать жест  [Пробел]'
    strings.tr("overlay_recording", left=1.5)      # str.format placeholders

The interface language is a user setting (settings.json -> "language"); the recognized words
themselves always come from data/vocab.csv in all three languages. All texts are exported for
review with scripts/export_ui_strings.py -> reports/ui_strings.csv.
"""

import csv
import string

from asl import config

LANGUAGES = config.UI_LANGUAGES

# key: {"en": ..., "hy": ..., "ru": ...}; "&" marks a menu shortcut letter.
STRINGS = {
    "app_title": {"en": "SLT", "hy": "SLT", "ru": "SLT"},
    "tab_single": {"en": "Single word", "hy": "Մեկ բառ", "ru": "Одно слово"},
    "tab_sentence": {"en": "Sentence", "hy": "Նախադասություն", "ru": "Предложение"},
    "tab_learn": {"en": "Learn", "hy": "Ուսուցում", "ru": "Обучение"},
    "about_title": {"en": "About SLT", "hy": "SLT ծրագրի մասին", "ru": "О программе SLT"},
    "about_text": {
        "en": "Recognizes {n_signs} isolated signs of American Sign Language (ASL) from the webcam and shows them in English, Armenian and Russian.\n\nModel: {model}. Accuracy on people it has never seen: top-1 {top1}, top-3 {top3}.\n\nData: Google — Isolated Sign Language Recognition (Kaggle).",
        "hy": "Տեսախցիկի միջոցով ճանաչում է ամերիկյան ժեստերի լեզվի (ASL) {n_signs} առանձին ժեստ և ցույց է տալիս դրանք անգլերեն, հայերեն և ռուսերեն։\n\nՄոդել՝ {model}։ Ճշգրտությունը մարդկանց վրա, որոնց մոդելը չի տեսել՝ top-1 {top1}, top-3 {top3}։\n\nՏվյալներ՝ Google — Isolated Sign Language Recognition (Kaggle)։",
        "ru": "Распознаёт {n_signs} отдельных жестов американского жестового языка (ASL) с веб-камеры и показывает их на английском, армянском и русском.\n\nМодель: {model}. Точность на людях, которых модель не видела: top-1 {top1}, top-3 {top3}.\n\nДанные: Google — Isolated Sign Language Recognition (Kaggle).",
    },
    "btn_record": {"en": "Record sign", "hy": "Տեսագրել ժեստը", "ru": "Записать жест"},
    "btn_speak": {"en": "Speak", "hy": "Արտասանել", "ru": "Произнести"},
    "overlay_get_ready": {"en": "Get ready", "hy": "Պատրաստվեք", "ru": "Приготовьтесь"},
    "overlay_recording": {
        "en": "Recording {left:.1f} s",
        "hy": "Տեսագրում՝ {left:.1f} վ",
        "ru": "Запись: {left:.1f} с",
    },
    "result_empty": {
        "en": "Press Space and show a sign: the word will appear here.",
        "hy": "Սեղմիր Բացատ և ցույց տուր ժեստը․ բառը կհայտնվի այստեղ։",
        "ru": "Нажми Пробел и покажи жест — слово появится здесь.",
    },
    "result_other": {"en": "Other guesses", "hy": "Այլ տարբերակներ", "ru": "Другие варианты"},
    "hint_camera_missing": {"en": "Camera not found", "hy": "Տեսախցիկը չի գտնվել", "ru": "Камера не найдена"},
    "hint_camera_starting": {"en": "Starting the camera…", "hy": "Տեսախցիկը միանում է…", "ru": "Камера запускается…"},
    "error_title": {"en": "Error", "hy": "Սխալ", "ru": "Ошибка"},
    "error_camera": {
        "en": "Camera {index} could not be opened or stopped working.\nCheck that it is connected and not used by another program, or choose another camera in Settings.\n\nDetails: {details}",
        "hy": "Տեսախցիկ {index}-ը չհաջողվեց միացնել, կամ այն դադարեց աշխատել։\nՍտուգեք, որ այն միացված է և չի օգտագործվում այլ ծրագրի կողմից, կամ Կարգավորումներում ընտրեք այլ տեսախցիկ։\n\nՄանրամասներ՝ {details}",
        "ru": "Не удалось открыть камеру {index}, или она перестала работать.\nПроверьте, что она подключена и не занята другой программой, или выберите другую камеру в настройках.\n\nПодробности: {details}",
    },
    "error_model": {
        "en": "The recognition model could not be loaded, so signs cannot be recognized.\n\nDetails: {details}",
        "hy": "Ճանաչման մոդելը չհաջողվեց բեռնել, ուստի ժեստերը չեն ճանաչվի։\n\nՄանրամասներ՝ {details}",
        "ru": "Не удалось загрузить модель распознавания, поэтому жесты не распознаются.\n\nПодробности: {details}",
    },
    "error_vocab": {
        "en": "The word list (data/vocab.csv) could not be loaded.\n\nDetails: {details}",
        "hy": "Բառացանկը (data/vocab.csv) չհաջողվեց բեռնել։\n\nՄանրամասներ՝ {details}",
        "ru": "Не удалось загрузить список слов (data/vocab.csv).\n\nПодробности: {details}",
    },
    "error_internal": {
        "en": "Something went wrong: {details}\nThe details are in logs/app.log.",
        "hy": "Ինչ-որ բան սխալ գնաց՝ {details}\nՄանրամասները logs/app.log ֆայլում են։",
        "ru": "Что-то пошло не так: {details}\nПодробности в logs/app.log.",
    },
    "error_speech": {
        "en": "Speech is not available: {details}",
        "hy": "Արտասանությունը հասանելի չէ՝ {details}",
        "ru": "Озвучивание недоступно: {details}",
    },
    "sent_empty": {
        "en": "The words will appear here.",
        "hy": "Բառերը կհայտնվեն այստեղ։",
        "ru": "Здесь появятся слова.",
    },
    "sent_manual": {
        "en": "Record each word with the button",
        "hy": "Յուրաքանչյուր բառը տեսագրել կոճակով",
        "ru": "Записывать каждое слово по кнопке",
    },
    "sent_alternatives": {
        "en": "Wrong word? Tap the right one:",
        "hy": "Սխա՞լ բառ է։ Սեղմիր ճիշտը՝",
        "ru": "Не то слово? Нажми правильное:",
    },
    "sent_delete_last": {"en": "Delete last", "hy": "Ջնջել վերջինը", "ru": "Удалить последнее"},
    "sent_clear": {"en": "Clear", "hy": "Մաքրել", "ru": "Очистить"},
    "sent_speak": {"en": "Speak", "hy": "Արտասանել", "ru": "Произнести"},
    "sent_copy": {"en": "Copy", "hy": "Պատճենել", "ru": "Копировать"},
    "sent_copied": {
        "en": "The sentence is copied.",
        "hy": "Նախադասությունը պատճենված է։",
        "ru": "Предложение скопировано.",
    },
    "sent_not_recognized": {
        "en": "The last sign was not recognized, try again.",
        "hy": "Վերջին ժեստը չճանաչվեց, փորձիր նորից։",
        "ru": "Последний жест не распознан, попробуй ещё раз.",
    },
    "learn_filter_all": {"en": "All", "hy": "Բոլորը", "ru": "Все"},
    "learn_filter_learned": {"en": "Learned", "hy": "Սովորած", "ru": "Выученные"},
    "learn_filter_new": {"en": "Not learned", "hy": "Չսովորած", "ru": "Не выученные"},
    "learn_reference": {"en": "Reference", "hy": "Օրինակ", "ru": "Эталон"},
    "learn_you": {"en": "Your attempt", "hy": "Քո փորձը", "ru": "Твоя попытка"},
    "learn_try": {"en": "Try", "hy": "Փորձել", "ru": "Попробовать"},
    "learn_pick": {"en": "Choose a sign in the list", "hy": "Ընտրիր ժեստը ցուցակից", "ru": "Выбери слово в списке"},
    "learn_result_ok": {"en": "Correct! {word} — {p}", "hy": "Ճիշտ է։ {word}՝ {p}", "ru": "Верно! {word} — {p}"},
    "learn_result_low": {
        "en": "{word} is first, but only {p} (needs {threshold}). Try once more.",
        "hy": "{word}-ը առաջինն է, բայց միայն {p} (պետք է {threshold})։ Փորձիր ևս մեկ անգամ։",
        "ru": "{word} на первом месте, но только {p} (нужно {threshold}). Попробуй ещё раз.",
    },
    "learn_result_other": {
        "en": "Looks like {other} ({p_other}). {word}: {p}, place {rank}",
        "hy": "Նման է {other}-ին ({p_other})։ {word}՝ {p}, {rank}-րդ տեղ",
        "ru": "Похоже на {other} ({p_other}). {word}: {p}, {rank}-е место",
    },
    "learn_result_none": {
        "en": "No hand was seen in the recording, try again",
        "hy": "Տեսագրության մեջ ձեռքը չի երևացել, փորձիր նորից",
        "ru": "В записи не видно руки, попробуй ещё раз",
    },
    "learn_progress": {
        "en": "Successful attempts: {n} of {need}",
        "hy": "Հաջող փորձեր՝ {n}-ը {need}-ից",
        "ru": "Удачных попыток: {n} из {need}",
    },
    "learn_learned": {"en": "Learned", "hy": "Սովորած", "ru": "Выучено"},
    "learn_export": {
        "en": "Export the defense set",
        "hy": "Արտահանել պաշտպանության հավաքածուն",
        "ru": "Экспортировать набор для защиты",
    },
    "learn_exported": {
        "en": "{n} signs saved to {path}",
        "hy": "{n} ժեստ պահպանվեց՝ {path}",
        "ru": "Сохранено жестов: {n} — {path}",
    },
    "learn_no_reference": {
        "en": "No reference for this sign.",
        "hy": "Այս ժեստի համար օրինակ չկա։",
        "ru": "Для этого жеста нет эталона.",
    },
    "error_references": {
        "en": "The reference clips (assets/reference) could not be loaded, so Learn mode is not available.\n\nDetails: {details}",
        "hy": "Օրինակների ֆայլերը (assets/reference) չհաջողվեց բեռնել, ուստի Ուսուցման ռեժիմը հասանելի չէ։\n\nՄանրամասներ՝ {details}",
        "ru": "Не удалось загрузить эталоны (assets/reference), поэтому режим обучения недоступен.\n\nПодробности: {details}",
    },
    "settings_title": {"en": "Settings", "hy": "Կարգավորումներ", "ru": "Настройки"},
    "settings_camera": {"en": "Camera number", "hy": "Տեսախցիկի համարը", "ru": "Номер камеры"},
    "settings_record_seconds": {"en": "Recording length", "hy": "Տեսագրման տևողությունը", "ru": "Длительность записи"},
    "settings_seconds_suffix": {"en": " s", "hy": " վ", "ru": " с"},
    "settings_threshold": {"en": "Confidence threshold", "hy": "Վստահության շեմը", "ru": "Порог уверенности"},
    "settings_sound": {
        "en": "Speak the recognized word",
        "hy": "Արտասանել ճանաչված բառը",
        "ru": "Произносить распознанное слово",
    },
    "settings_mirror": {"en": "Mirror the video", "hy": "Հայելային պատկեր", "ru": "Зеркальное изображение"},
    "settings_language": {"en": "Interface language", "hy": "Ինտերֆեյսի լեզուն", "ru": "Язык интерфейса"},
    "btn_ok": {"en": "OK", "hy": "Լավ", "ru": "ОК"},
    "btn_cancel": {"en": "Cancel", "hy": "Չեղարկել", "ru": "Отмена"},
    "lang_name_hy": {"en": "Հայերեն", "hy": "Հայերեն", "ru": "Հայերեն"},
    "lang_name_ru": {"en": "Русский", "hy": "Русский", "ru": "Русский"},
    "lang_name_en": {"en": "English", "hy": "English", "ru": "English"},
    "tip_settings": {"en": "Settings", "hy": "Կարգավորումներ", "ru": "Настройки"},
    "tip_help": {"en": "About the program", "hy": "Ծրագրի մասին", "ru": "О программе"},
    "key_space": {"en": "Space", "hy": "Բացատ", "ru": "Пробел"},
    "hint_space": {
        "en": "Press {key} and show a sign",
        "hy": "Սեղմիր {key} և ցույց տուր ժեստը",
        "ru": "Нажми {key} и покажи жест",
    },
    "hint_lower_hand": {
        "en": "Lower your hand between signs",
        "hy": "Ժեստերի միջև իջեցրու ձեռքը",
        "ru": "Между жестами опускай руку",
    },
    "chip_fps": {"en": "{fps} fps", "hy": "{fps} կ/վ", "ru": "{fps} к/с"},
    "chip_shoulders_ok": {"en": "Shoulders visible", "hy": "Ուսերը երևում են", "ru": "Плечи видны"},
    "chip_hand_ok": {"en": "Hand visible", "hy": "Ձեռքը երևում է", "ru": "Рука видна"},
    "chip_shoulders_missing": {
        "en": "Shoulders not visible — move back",
        "hy": "Ուսերը չեն երևում, մի փոքր հետ գնա",
        "ru": "Плечи не видны — отодвинься",
    },
    "chip_hand_missing": {"en": "Hand not visible", "hy": "Ձեռքը չի երևում", "ru": "Рука не видна"},
    "seg_idle": {"en": "Waiting", "hy": "Սպասում", "ru": "Ожидание"},
    "seg_sign": {"en": "Sign", "hy": "Ժեստ", "ru": "Жест"},
    "seg_pause": {"en": "Pause", "hy": "Դադար", "ru": "Пауза"},
    "res_recognized": {"en": "Recognized", "hy": "Ճանաչվեց", "ru": "Распознано"},
    "res_not_recognized": {
        "en": "Not recognized — try again",
        "hy": "Չճանաչվեց, փորձիր նորից",
        "ru": "Не распознано — попробуй ещё раз",
    },
    "res_confidence": {"en": "Confidence", "hy": "Վստահություն", "ru": "Уверенность"},
    "sent_manual_note": {
        "en": "More reliable on stage: each word with Space",
        "hy": "Ելույթի ժամանակ ավելի հուսալի է՝ յուրաքանչյուր բառ Բացատով",
        "ru": "Надёжнее на выступлении: каждое слово — по Пробелу",
    },
    "learn_search_label": {"en": "Find a sign", "hy": "Ժեստի որոնում", "ru": "Поиск слова"},
    "learn_search_placeholder": {"en": "dad, папа, հայրիկ…", "hy": "dad, папа, հայրիկ…", "ru": "dad, папа, հայրիկ…"},
    "learn_count": {
        "en": "Learned: {n} of {total}",
        "hy": "Սովորած՝ {n}-ը {total}-ից",
        "ru": "Выучено: {n} из {total}",
    },
    "learn_sync": {
        "en": "Reference and attempt play in sync",
        "hy": "Օրինակն ու փորձը ընթանում են համաժամ",
        "ru": "Эталон и попытка идут синхронно",
    },
    "learn_ready": {
        "en": "Watch the reference, then press «Try»",
        "hy": "Դիտիր օրինակը, ապա սեղմիր «Փորձել»",
        "ru": "Посмотри эталон и нажми «Попробовать»",
    },
    "learn_progress_one_left": {
        "en": "Successful attempts: {n} of {need} — one more and the sign is learned",
        "hy": "Հաջող փորձեր՝ {n}-ը {need}-ից։ Եվս մեկը, և ժեստը սովորած է",
        "ru": "Удачных попыток: {n} из {need} — ещё одна, и слово выучено",
    },
    "learn_progress_done": {
        "en": "Learned! Successful attempts: {n}",
        "hy": "Սովորած է։ Հաջող փորձեր՝ {n}",
        "ru": "Выучено! Удачных попыток: {n}",
    },
    "settings_show_camera": {
        "en": "Show the camera picture under the skeleton",
        "hy": "Ցույց տալ տեսախցիկի պատկերը կմախքի տակ",
        "ru": "Показывать изображение камеры под скелетом",
    },
}

_language = LANGUAGES[0]


def set_language(lang: str) -> None:
    global _language
    if lang not in LANGUAGES:
        raise ValueError(f"unknown interface language {lang!r}; known: {', '.join(LANGUAGES)}")
    _language = lang


def language() -> str:
    return _language


def tr(key: str, **fmt) -> str:
    """Text of `key` in the current interface language, formatted with `fmt`."""
    text = STRINGS[key][_language]
    return text.format(**fmt) if fmt else text


def placeholders(text: str) -> set:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def export_csv(path=config.UI_STRINGS_CSV) -> int:
    """All strings as key, en, hy, ru (for the user's review); returns the number of rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # BOM: Excel opens it as UTF-8
        w = csv.writer(f)
        w.writerow(["key", "en", "hy", "ru"])
        for key, texts in STRINGS.items():
            w.writerow([key, texts["en"], texts["hy"], texts["ru"]])
    return len(STRINGS)
