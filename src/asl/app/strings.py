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
    "app_title": {"en": "ASL Translator", "hy": "ASL թարգմանիչ", "ru": "Переводчик ASL"},
    "tab_single": {"en": "Single word", "hy": "Մեկ բառ", "ru": "Одно слово"},
    "tab_sentence": {"en": "Sentence", "hy": "Նախադասություն", "ru": "Предложение"},
    "tab_learn": {"en": "Learn", "hy": "Ուսուցում", "ru": "Обучение"},
    "tab_coming_soon": {
        "en": "This mode will be added in a later version.",
        "hy": "Այս ռեժիմը կավելացվի հաջորդ տարբերակներում։",
        "ru": "Этот режим появится в следующих версиях.",
    },
    "menu_file": {"en": "&File", "hy": "&Ֆայլ", "ru": "&Файл"},
    "menu_exit": {"en": "E&xit", "hy": "&Ելք", "ru": "&Выход"},
    "menu_settings": {"en": "&Settings", "hy": "&Կարգավորումներ", "ru": "&Настройки"},
    "menu_settings_open": {"en": "&Settings…", "hy": "&Կարգավորումներ…", "ru": "&Настройки…"},
    "menu_help": {"en": "&Help", "hy": "&Օգնություն", "ru": "&Справка"},
    "menu_about": {"en": "&About", "hy": "&Ծրագրի մասին", "ru": "&О программе"},
    "about_title": {"en": "About ASL Translator", "hy": "ASL թարգմանչի մասին", "ru": "О программе «Переводчик ASL»"},
    "about_text": {
        "en": "Recognizes {n_signs} isolated signs of American Sign Language (ASL) from the webcam "
              "and shows them in English, Armenian and Russian.\n\n"
              "Model: {model}. Accuracy on people it has never seen: top-1 {top1}, top-3 {top3}.\n\n"
              "Data: Google — Isolated Sign Language Recognition (Kaggle).",
        "hy": "Տեսախցիկի միջոցով ճանաչում է ամերիկյան ժեստերի լեզվի (ASL) {n_signs} առանձին ժեստ "
              "և ցույց է տալիս դրանք անգլերեն, հայերեն և ռուսերեն։\n\n"
              "Մոդել՝ {model}։ Ճշգրտությունը մարդկանց վրա, որոնց մոդելը չի տեսել՝ top-1 {top1}, top-3 {top3}։\n\n"
              "Տվյալներ՝ Google — Isolated Sign Language Recognition (Kaggle)։",
        "ru": "Распознаёт {n_signs} отдельных жестов американского жестового языка (ASL) с веб-камеры "
              "и показывает их на английском, армянском и русском.\n\n"
              "Модель: {model}. Точность на людях, которых модель не видела: top-1 {top1}, top-3 {top3}.\n\n"
              "Данные: Google — Isolated Sign Language Recognition (Kaggle).",
    },
    "btn_record": {"en": "Record sign  [Space]", "hy": "Տեսագրել ժեստը  [Բացատ]", "ru": "Записать жест  [Пробел]"},
    "btn_speak": {"en": "Speak", "hy": "Արտասանել", "ru": "Произнести"},
    "hint_press_record": {
        "en": "Press Space or «Record sign», then show one sign after the countdown.",
        "hy": "Սեղմեք Բացատ կամ «Տեսագրել ժեստը», ապա հետհաշվարկից հետո ցույց տվեք մեկ ժեստ։",
        "ru": "Нажмите пробел или «Записать жест» и после отсчёта покажите один жест.",
    },
    "overlay_get_ready": {"en": "Get ready", "hy": "Պատրաստվեք", "ru": "Приготовьтесь"},
    "overlay_recording": {"en": "Recording {left:.1f} s", "hy": "Տեսագրում՝ {left:.1f} վ", "ru": "Запись: {left:.1f} с"},
    "overlay_processing": {"en": "Recognizing…", "hy": "Ճանաչվում է…", "ru": "Распознаю…"},
    "overlay_fps": {"en": "FPS {fps:.0f}", "hy": "FPS {fps:.0f}", "ru": "FPS {fps:.0f}"},
    "result_empty": {"en": "The result will appear here.", "hy": "Արդյունքը կհայտնվի այստեղ։",
                     "ru": "Здесь появится результат."},
    "result_not_recognized": {"en": "Not recognized, try again", "hy": "Չհաջողվեց ճանաչել, փորձեք նորից",
                              "ru": "Не распознано, попробуйте ещё раз"},
    "result_best_guess": {"en": "Closest: {word} ({p})", "hy": "Ամենամոտը՝ {word} ({p})", "ru": "Ближе всего: {word} ({p})"},
    "result_confidence": {"en": "Confidence: {p}", "hy": "Վստահություն՝ {p}", "ru": "Уверенность: {p}"},
    "result_other": {"en": "Other guesses", "hy": "Այլ տարբերակներ", "ru": "Другие варианты"},
    "hint_no_hands": {"en": "Hands not visible", "hy": "Ձեռքերը չեն երևում", "ru": "Руки не видны"},
    "hint_no_shoulders": {"en": "Shoulders not visible — move back", "hy": "Ուսերը չեն երևում, մի փոքր հետ գնացեք",
                          "ru": "Плечи не видны — отодвиньтесь"},
    "hint_camera_missing": {"en": "Camera not found", "hy": "Տեսախցիկը չի գտնվել", "ru": "Камера не найдена"},
    "hint_camera_starting": {"en": "Starting the camera…", "hy": "Տեսախցիկը միանում է…", "ru": "Камера запускается…"},
    "error_title": {"en": "Error", "hy": "Սխալ", "ru": "Ошибка"},
    "error_camera": {
        "en": "Camera {index} could not be opened or stopped working.\n"
              "Check that it is connected and not used by another program, or choose another camera in Settings.\n\n"
              "Details: {details}",
        "hy": "Տեսախցիկ {index}-ը չհաջողվեց միացնել, կամ այն դադարեց աշխատել։\n"
              "Ստուգեք, որ այն միացված է և չի օգտագործվում այլ ծրագրի կողմից, կամ Կարգավորումներում ընտրեք այլ տեսախցիկ։\n\n"
              "Մանրամասներ՝ {details}",
        "ru": "Не удалось открыть камеру {index}, или она перестала работать.\n"
              "Проверьте, что она подключена и не занята другой программой, или выберите другую камеру в настройках.\n\n"
              "Подробности: {details}",
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
    "error_speech": {"en": "Speech is not available: {details}", "hy": "Արտասանությունը հասանելի չէ՝ {details}",
                     "ru": "Озвучивание недоступно: {details}"},
    "sent_empty": {"en": "The words will appear here.", "hy": "Բառերը կհայտնվեն այստեղ։", "ru": "Здесь появятся слова."},
    "sent_lang_label": {"en": "Show the words in:", "hy": "Բառերը ցույց տալ՝", "ru": "Показывать слова на:"},
    "sent_manual": {"en": "Record each word with the button", "hy": "Յուրաքանչյուր բառը տեսագրել կոճակով",
                    "ru": "Записывать каждое слово по кнопке"},
    "sent_alternatives": {"en": "Replace the last word:", "hy": "Փոխարինել վերջին բառը՝",
                          "ru": "Заменить последнее слово:"},
    "sent_delete_last": {"en": "Delete last", "hy": "Ջնջել վերջինը", "ru": "Удалить последнее"},
    "sent_clear": {"en": "Clear", "hy": "Մաքրել", "ru": "Очистить"},
    "sent_speak": {"en": "Speak sentence", "hy": "Արտասանել նախադասությունը", "ru": "Произнести предложение"},
    "sent_copy": {"en": "Copy", "hy": "Պատճենել", "ru": "Копировать"},
    "sent_copied": {"en": "The sentence is copied.", "hy": "Նախադասությունը պատճենված է։",
                    "ru": "Предложение скопировано."},
    "sent_not_recognized": {"en": "The last sign was not recognized, try again.",
                            "hy": "Վերջին ժեստը չճանաչվեց, փորձեք նորից։",
                            "ru": "Последний жест не распознан, попробуйте ещё раз."},
    "sent_help_auto": {
        "en": "Lower your hands between signs: a sign starts when the hand goes up and ends when it goes down.",
        "hy": "Ժեստերի միջև իջեցրեք ձեռքերը։ Ժեստը սկսվում է, երբ ձեռքը բարձրանում է, և ավարտվում, երբ իջնում է։",
        "ru": "Между жестами опускайте руки: жест начинается, когда рука поднимается, и заканчивается, когда она опускается.",
    },
    "sent_help_manual": {
        "en": "For each word press Space or «Record sign»; the words are added to the line.",
        "hy": "Յուրաքանչյուր բառի համար սեղմեք Բացատ կամ «Տեսագրել ժեստը», բառերը կավելանան տողին։",
        "ru": "Для каждого слова нажмите пробел или «Записать жест» — слова добавятся в строку.",
    },
    "overlay_seg_idle": {"en": "Waiting — raise your hand", "hy": "Սպասում՝ բարձրացրեք ձեռքը",
                         "ru": "Ожидание — поднимите руку"},
    "overlay_seg_sign": {"en": "Sign", "hy": "Ժեստ", "ru": "Жест"},
    "overlay_seg_pause": {"en": "Pause", "hy": "Դադար", "ru": "Пауза"},
    "settings_title": {"en": "Settings", "hy": "Կարգավորումներ", "ru": "Настройки"},
    "settings_camera": {"en": "Camera number", "hy": "Տեսախցիկի համարը", "ru": "Номер камеры"},
    "settings_record_seconds": {"en": "Recording length", "hy": "Տեսագրման տևողությունը", "ru": "Длительность записи"},
    "settings_seconds_suffix": {"en": " s", "hy": " վ", "ru": " с"},
    "settings_threshold": {"en": "Confidence threshold", "hy": "Վստահության շեմը", "ru": "Порог уверенности"},
    "settings_sound": {"en": "Speak the recognized word", "hy": "Արտասանել ճանաչված բառը",
                       "ru": "Произносить распознанное слово"},
    "settings_mirror": {"en": "Mirror the video", "hy": "Հայելային պատկեր", "ru": "Зеркальное изображение"},
    "settings_language": {"en": "Interface language", "hy": "Ինտերֆեյսի լեզուն", "ru": "Язык интерфейса"},
    "btn_ok": {"en": "OK", "hy": "Լավ", "ru": "ОК"},
    "btn_cancel": {"en": "Cancel", "hy": "Չեղարկել", "ru": "Отмена"},
    # language names are written in the language itself in every interface language
    "lang_name_hy": {"en": "Հայերեն", "hy": "Հայերեն", "ru": "Հայերեն"},
    "lang_name_ru": {"en": "Русский", "hy": "Русский", "ru": "Русский"},
    "lang_name_en": {"en": "English", "hy": "English", "ru": "English"},
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
