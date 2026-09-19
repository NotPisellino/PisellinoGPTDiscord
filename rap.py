import random
import tempfile
from pathlib import Path

import librosa
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range, normalize
from pydub.silence import detect_nonsilent


# ============================================================
# CONFIG
# ============================================================

SAMPLES_DIR = Path("samples")
OUTPUT_FILE = Path("pisellino_rap_v3.mp3")

BPM = 140
DURATION = 30
SR = 44100

# Quanto aggressivo deve essere lo snapping ritmico.
# 1.0 = molto preciso
# 0.7 = più naturale
RHYTHM_STRENGTH = 0.85


# ============================================================
# LYRICS
# ============================================================

LYRICS = """
Domenico parla tanto ma non sa giocare
entra nel server e lo vediamo laggare
dice che è forte ma dove vuole andare
fa tre passi avanti e poi torna a spawnare

Domenico bro ma che stai combinando
io faccio una kill mentre tu stai quittando
dici sono forte continua pure a parlare
quando parte questo beat ti vediamo volare

non è cattiveria è soltanto intrattenimento
ma se giochi così ti serve allenamento
entra nella lobby preparati al disastro
parte questo beat e Domenico fa il clown
""".strip()


# ============================================================
# AD-LIBS
# ============================================================

# gTTS italiano tende a leggere le parole inglesi
# letteralmente. Usiamo quindi versioni fonetiche italiane.

ADLIBS = [
    "iea",
    "ei",
    "ah",
    "ah ah",
    "uò",
    "iea iea",
    "aah",
]

# Molto più bassi rispetto alla voce principale.
ADLIB_GAIN = -14

# Possibilità di mettere un ad-lib su una posizione valida.
ADLIB_CHANCE = 0.30


# ============================================================
# SAMPLE FINDER
# ============================================================

def find_sample(*keywords):
    files = list(SAMPLES_DIR.glob("*"))

    for keyword in keywords:
        keyword = keyword.lower()

        for file in files:
            if keyword in file.stem.lower():
                return file

    return None


# ============================================================
# AUDIO HELPERS
# ============================================================

def load_audio(path):
    audio, _ = librosa.load(
        path,
        sr=SR,
        mono=True
    )

    return audio


def audiosegment_from_numpy(audio):
    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    audio = np.clip(
        audio,
        -1.0,
        1.0
    )

    pcm = (
        audio * 32767
    ).astype(np.int16)

    return AudioSegment(
        pcm.tobytes(),
        frame_rate=SR,
        sample_width=2,
        channels=1
    )


def numpy_from_segment(segment):
    segment = (
        segment
        .set_frame_rate(SR)
        .set_channels(1)
        .set_sample_width(2)
    )

    samples = np.array(
        segment.get_array_of_samples()
    ).astype(np.float32)

    samples /= 32768.0

    return np.nan_to_num(
        samples
    )


# ============================================================
# PITCH SHIFT
# ============================================================

def pitch_shift(segment, semitones):
    """
    Pitch shifting sicuro.
    Evita di processare segmenti microscopici.
    """

    if len(segment) < 180:
        return segment

    if abs(semitones) < 0.05:
        return segment

    audio = numpy_from_segment(
        segment
    )

    if len(audio) < 512:
        return segment

    try:
        shifted = librosa.effects.pitch_shift(
            audio,
            sr=SR,
            n_steps=semitones
        )

        return audiosegment_from_numpy(
            shifted
        )

    except Exception as e:
        print(
            f"[PITCH] Failed: {e}"
        )

        return segment


# ============================================================
# TIME STRETCH
# ============================================================

def stretch_to(segment, target_ms):
    """
    Cambia la velocità senza cambiare il pitch.
    """

    if len(segment) < 180:
        return segment

    if target_ms <= 0:
        return segment

    ratio = (
        len(segment)
        / target_ms
    )

    # Protezione contro trasformazioni mostruose.
    ratio = max(
        0.70,
        min(
            ratio,
            1.40
        )
    )

    if abs(ratio - 1.0) < 0.025:
        return segment

    audio = numpy_from_segment(
        segment
    )

    try:
        stretched = librosa.effects.time_stretch(
            audio,
            rate=ratio
        )

        return audiosegment_from_numpy(
            stretched
        )

    except Exception as e:
        print(
            f"[STRETCH] Failed: {e}"
        )

        return segment


# ============================================================
# TTS
# ============================================================

def generate_tts(text):
    print(
        f'[TTS] "{text}"'
    )

    temp = Path(
        tempfile.mktemp(
            suffix=".mp3"
        )
    )

    tts = gTTS(
        text=text,
        lang="it",
        slow=False
    )

    tts.save(
        str(temp)
    )

    audio = AudioSegment.from_file(
        temp
    )

    temp.unlink(
        missing_ok=True
    )

    return audio


# ============================================================
# REVERB
# ============================================================

def add_reverb(audio):
    wet = AudioSegment.silent(
        duration=len(audio) + 500
    )

    for delay, gain in [
        (70, -18),
        (135, -22),
        (220, -26),
        (330, -30),
    ]:
        copy = audio.apply_gain(
            gain
        )

        wet = wet.overlay(
            copy,
            position=delay
        )

    return audio.overlay(
        wet
    )


# ============================================================
# VOCAL MASTERING
# ============================================================

def process_main_voice(voice):
    voice = compress_dynamic_range(
        voice,
        threshold=-20,
        ratio=3.5,
        attack=5,
        release=80
    )

    voice = voice.high_pass_filter(
        100
    )

    voice = voice.low_pass_filter(
        11500
    )

    voice = normalize(
        voice
    )

    return voice


# ============================================================
# PHONETIC / RHYTHMIC TEXT SPLITTING
# ============================================================

def split_into_rap_units(text):
    """
    Divide una riga in piccole unità pronunciate.

    Non è un vero fonetic aligner, ma crea unità abbastanza
    piccole da poterle distribuire sulla griglia.
    """

    words = text.split()

    units = []

    for word in words:

        clean = (
            word
            .replace(",", "")
            .replace(".", "")
            .replace("!", "")
            .replace("?", "")
            .replace(":", "")
            .replace(";", "")
        )

        if not clean:
            continue

        # Cerca di dividere parole lunghe.
        if len(clean) >= 9:

            # Approssimazione delle sillabe italiane.
            vowels = "aeiouàèéìòóù"

            pieces = []
            current = ""

            for char in clean:

                current += char

                if (
                    char.lower() in vowels
                    and len(current) >= 2
                ):
                    pieces.append(
                        current
                    )

                    current = ""

            if current:
                pieces.append(
                    current
                )

            # Se la divisione è sensata usala.
            if len(pieces) >= 2:

                # Evita pezzi ridicolmente piccoli.
                if all(
                    len(x) >= 2
                    for x in pieces
                ):
                    units.extend(
                        pieces
                    )
                    continue

        units.append(
            clean
        )

    return units


# ============================================================
# CREATE RHYTHMIC VOCALS
# ============================================================

def create_rhythmic_vocals(
    lyrics,
    bpm
):
    print(
        "[FLOW] Building syllable/word rhythm..."
    )

    beat_ms = (
        60000 / bpm
    )

    # Sedicesimo.
    subdivision = beat_ms / 4

    # Generiamo ogni riga separatamente.
    lines = [
        line.strip()
        for line in lyrics.splitlines()
        if line.strip()
    ]

    final = AudioSegment.empty()

    for line_index, line in enumerate(lines):

        print(
            f'[FLOW] Line {line_index + 1}: "{line}"'
        )

        units = split_into_rap_units(
            line
        )

        if not units:
            continue

        # ---------------------------------------------
        # TTS DELLA RIGA
        # ---------------------------------------------

        line_audio = generate_tts(
            line
        )

        # ---------------------------------------------
        # Trova le parti sonore
        # ---------------------------------------------

        ranges = detect_nonsilent(
            line_audio,
            min_silence_len=35,
            silence_thresh=line_audio.dBFS - 20,
            seek_step=5
        )

        if not ranges:

            final += line_audio
            continue

        # ---------------------------------------------
        # Costruiamo i segmenti sonori
        # ---------------------------------------------

        chunks = []

        for start, end in ranges:

            start = max(
                0,
                start - 15
            )

            end = min(
                len(line_audio),
                end + 15
            )

            chunks.append(
                line_audio[start:end]
            )

        # ---------------------------------------------
        # Distribuzione ritmica
        # ---------------------------------------------

        current_position = 0

        for i, chunk in enumerate(chunks):

            original = len(
                chunk
            )

            # La durata ideale è vicina a una
            # suddivisione ritmica.
            subdivisions_needed = max(
                1,
                round(
                    original
                    / subdivision
                )
            )

            target = (
                subdivisions_needed
                * subdivision
            )

            # Interpolazione verso la durata originale
            target = (
                original
                * (1 - RHYTHM_STRENGTH)
                + target
                * RHYTHM_STRENGTH
            )

            target = max(
                90,
                target
            )

            # -----------------------------------------
            # Pitch variation
            # -----------------------------------------

            pitch = 0

            # Non cambiare ogni pezzo.
            if random.random() < 0.22:

                pitch = random.choice(
                    [
                        -2,
                        -1,
                        1,
                        2,
                    ]
                )

            # Fine della frase = leggero accent.
            if i == len(chunks) - 1:

                pitch += random.choice(
                    [-1, 0, 1]
                )

            if pitch:

                print(
                    f"[FLOW] "
                    f"chunk {i}: "
                    f"{pitch:+} semitones"
                )

                chunk = pitch_shift(
                    chunk,
                    pitch
                )

            # -----------------------------------------
            # Time stretch
            # -----------------------------------------

            chunk = stretch_to(
                chunk,
                int(target)
            )

            # -----------------------------------------
            # Piccolo micro-gap
            # -----------------------------------------

            final += chunk

            # Gap ritmico
            if i < len(chunks) - 1:

                if i % 4 == 3:
                    gap = subdivision * 0.45
                else:
                    gap = subdivision * 0.08

                final += AudioSegment.silent(
                    duration=int(gap)
                )

        # Fine riga = pausa più grande.
        final += AudioSegment.silent(
            duration=int(
                beat_ms * 0.35
            )
        )

    return final


# ============================================================
# AD-LIB GENERATION
# ============================================================

def generate_adlibs():
    print(
        "[ADLIB] Generating quiet ad-libs..."
    )

    result = []

    # Pochi, per evitare di riempire tutto.
    for i in range(8):

        if random.random() > 0.75:
            continue

        text = random.choice(
            ADLIBS
        )

        audio = generate_tts(
            text
        )

        # Pitch più evidente sugli ad-lib.
        pitch = random.choice(
            [
                -3,
                -2,
                2,
                3,
                4,
            ]
        )

        audio = pitch_shift(
            audio,
            pitch
        )

        audio = compress_dynamic_range(
            audio,
            threshold=-25,
            ratio=4,
            attack=5,
            release=80
        )

        audio = normalize(
            audio
        )

        audio = add_reverb(
            audio
        )

        # MOLTO più quiet.
        audio = audio.apply_gain(
            ADLIB_GAIN
        )

        # Stereo position.
        audio = audio.pan(
            random.uniform(
                -0.55,
                0.55
            )
        )

        result.append(
            audio
        )

        print(
            f"[ADLIB] {text} "
            f"({pitch:+} semitones, "
            f"{ADLIB_GAIN} dB)"
        )

    return result


# ============================================================
# PLACE AD-LIBS
# ============================================================

def place_adlibs(
    vocals,
    adlibs,
    bpm
):
    if not adlibs:
        return vocals

    print(
        "[ADLIB] Placing ad-libs..."
    )

    result = vocals

    beat_ms = (
        60000 / bpm
    )

    # Le posizioni possibili sono sugli off-beats.
    positions = []

    step = beat_ms / 2

    position = step

    while position < len(result):

        if random.random() < ADLIB_CHANCE:

            positions.append(
                int(position)
            )

        position += step

    random.shuffle(
        positions
    )

    for i, adlib in enumerate(
        adlibs
    ):

        if i >= len(positions):
            break

        position = positions[i]

        if position >= len(result):
            continue

        available = (
            len(result)
            - position
        )

        if len(adlib) > available:

            adlib = adlib[
                :available
            ]

        result = result.overlay(
            adlib,
            position=position
        )

    return result


# ============================================================
# MIX
# ============================================================

def mix(
    beat,
    vocals
):
    print(
        "[MIX] Mixing instrumental + vocals..."
    )

    beat = beat.set_channels(
        2
    )

    vocals = vocals.set_channels(
        2
    )

    if len(vocals) < len(beat):

        vocals += AudioSegment.silent(
            duration=len(beat) - len(vocals)
        )

    else:

        vocals = vocals[
            :len(beat)
        ]

    # Beat sotto la voce.
    beat = beat.apply_gain(
        -3
    )

    vocals = vocals.apply_gain(
        1
    )

    final = beat.overlay(
        vocals
    )

    final = normalize(
        final
    )

    return final


# ============================================================
# MAIN
# ============================================================

# ============================================================
# DISCORD API
# ============================================================

def generate_rap(lyrics, output_file=None):
    """
    Genera un rap completo partendo dai lyrics.

    Restituisce il path del file MP3 generato.
    """

    if output_file is None:
        output_file = OUTPUT_FILE

    output_file = Path(output_file)

    print()
    print("========================================")
    print("       PISELLINOGPT RAP ENGINE V3")
    print("========================================")
    print()

    if not SAMPLES_DIR.exists():
        raise FileNotFoundError(
            f"Cartella '{SAMPLES_DIR}' non trovata."
        )

    # --------------------------------------------------------
    # BEAT
    # --------------------------------------------------------

    print("[BEAT] Building instrumental...")

    kick_path = find_sample("kick")
    snare_path = find_sample("snare")
    hihat_path = find_sample(
        "hihat",
        "hi-hat",
        "hat"
    )
    bass_path = find_sample(
        "bass",
        "808"
    )

    if not kick_path:
        raise FileNotFoundError("Kick non trovato.")

    if not snare_path:
        raise FileNotFoundError("Snare non trovato.")

    if not hihat_path:
        raise FileNotFoundError("Hi-hat non trovato.")

    if not bass_path:
        raise FileNotFoundError(
            "Bass/808 non trovato."
        )

    kick = load_audio(kick_path)
    snare = load_audio(snare_path)
    hihat = load_audio(hihat_path)
    bass = load_audio(bass_path)

    total_samples = int(
        DURATION * SR
    )

    beat = np.zeros(
        total_samples,
        dtype=np.float32
    )

    beat_length = 60.0 / BPM
    sixteenth = beat_length / 4

    bass = bass[:total_samples]

    if len(bass) < total_samples:
        bass = np.resize(
            bass,
            total_samples
        )

    beat += bass * 0.42

    def add(sample, time, volume):
        position = int(
            time * SR
        )

        if position >= total_samples:
            return

        end = min(
            position + len(sample),
            total_samples
        )

        length = end - position

        beat[position:end] += (
            sample[:length]
            * volume
        )

    bars = int(
        DURATION
        / (beat_length * 4)
    )

    for bar in range(bars):

        start = (
            bar
            * beat_length
            * 4
        )

        if bar % 4 == 3:
            kick_pattern = [
                0,
                1.0,
                2.0,
                2.5,
                3.5,
            ]
        else:
            kick_pattern = [
                0,
                1.5,
                2.0,
                3.0,
            ]

        for p in kick_pattern:
            add(
                kick,
                start + p * beat_length,
                0.90
            )

        add(
            snare,
            start + beat_length,
            0.75
        )

        add(
            snare,
            start + beat_length * 3,
            0.75
        )

        for i in range(16):

            if random.random() < 0.07:
                continue

            volume = (
                0.38
                if i % 4 == 0
                else 0.25
            )

            add(
                hihat,
                start + i * sixteenth,
                volume
            )

        if bar % 4 == 3:

            for i in range(8):

                add(
                    hihat,
                    start
                    + beat_length * 3.75
                    + i * (sixteenth / 2),
                    0.24
                )

    beat = np.tanh(
        beat * 1.15
    )

    peak = np.max(
        np.abs(beat)
    )

    if peak > 0:
        beat /= peak

    beat = audiosegment_from_numpy(
        beat
    )

    # --------------------------------------------------------
    # VOCALS
    # --------------------------------------------------------

    vocals = create_rhythmic_vocals(
        lyrics,
        BPM
    )

    vocals = process_main_voice(
        vocals
    )

    # --------------------------------------------------------
    # AD-LIBS
    # --------------------------------------------------------

    adlibs = generate_adlibs()

    vocals = place_adlibs(
        vocals,
        adlibs,
        BPM
    )

    # --------------------------------------------------------
    # MIX
    # --------------------------------------------------------

    final = mix(
        beat,
        vocals
    )

    print(
        f"[OUTPUT] Saving {output_file}..."
    )

    final.export(
        output_file,
        format="mp3",
        bitrate="192k"
    )

    print(
        f"[OUTPUT] Done: {output_file}"
    )

    return output_file


# ============================================================
# STANDALONE MODE
# ============================================================

if __name__ == "__main__":

    generate_rap(
        LYRICS,
        OUTPUT_FILE
    )