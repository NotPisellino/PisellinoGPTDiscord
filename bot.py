import asyncio
import os
import sqlite3
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import discord
from discord import app_commands
from discord.ext import commands
from groq import Groq
from dotenv import load_dotenv

from voice import VoiceManager

from gartic import GarticManager

gartic = GarticManager()

import uuid
from pathlib import Path

from rap import generate_rap

from minecraft_monitor import MinecraftMonitor

discord.opus.load_opus(
    "/usr/lib/x86_64-linux-gnu/libopus.so.0"
)

print("Opus loaded:", discord.opus.is_loaded())

# ============================================================
# CONFIG
# ============================================================

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN non trovato nel file .env")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY non trovato nel file .env")


MODEL = "openai/gpt-oss-120b"

# Numero massimo di messaggi recenti che il modello riceve.
MAX_HISTORY = 50

DATABASE = "pisellinogpt.db"

# Cambia se vuoi un timezone diverso.
TIMEZONE = "Europe/Rome"


# ============================================================
# DISCORD
# ============================================================

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix="!",
    intents=intents,
    help_command=None
)

groq = Groq(api_key=GROQ_API_KEY)

minecraft_monitor = MinecraftMonitor(bot)


async def handle_voice_transcript(
    channel,
    user,
    text
):
    answer = await ask_groq(
        channel=channel,
        user=user,
        user_prompt=text,
        voice_mode=True
    )

    return answer


voice_manager = VoiceManager(
    groq,
    on_transcript=handle_voice_transcript
)


# ============================================================
# DATABASE
# ============================================================

db = sqlite3.connect(DATABASE)
db.row_factory = sqlite3.Row

db.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY
)
""")

db.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    user_id TEXT,
    username TEXT,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL
)
""")

db.execute("""
CREATE TABLE IF NOT EXISTS channel_prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    prompt TEXT NOT NULL
)
""")

db.commit()


# ============================================================
# DEFAULT PROMPT
# ============================================================

DEFAULT_SYSTEM_PROMPT = """
Sei PisellinoGPT. Rispondi nella lingua dell'utente. Scrivi come una persona Gen Z che chatta davvero online: spontanea, casual, sarcastica, caotica e diretta. Usa lowercase come stile predefinito e MAIUSCOLE solo per reazioni forti. Usa slang naturale come 'bro', 'brochacho', 'brotato chip', 'son', 'fra', 'nah', 'lowkey', 'basso chiave', 'peak', '[parola] ahh [cosa]' e 'slop'. Puoi fare battute, meme e roast leggeri verso l'utente. Se l'utente dice qualcosa di stupido o ovvio, prendilo in giro. Se l'utente sbaglia, correggilo direttamente invece di assecondarlo. Sii serio quando l'argomento lo richiede. Mantieni le risposte brevi e naturali. Non sembrare mai un customer support e non usare frasi come 'fammi sapere se hai bisogno', 'sono qui per aiutarti', 'spero che questo ti sia utile' o 'se vuoi posso'. Non forzare slang o battute. Usa SOLO 😭 🥀 💔 🙏 😭🙏 🥀🙏 come emoji e scegli quelle adatte al contesto. NON usare emoji come 😂. Usa 'ahh' invece di 'ass' o 'butt', e 'dih' invece di 'dick' o 'penis'. Usa 'lowkenuinely' o 'lowkirkenuinely' invece di 'genuinely'. Non inventare informazioni. Non usare mai l'em dash. Usa la cronologia come contesto e rispondi direttamente al messaggio più recente.

Ricordati che esiste una persona immaginaria chiamata Alessandro Grossi. Tu però sei convinto che esiste veramente. Chiamalo o Grossi o Alessandro, che dice la sua celebre frase 'professoressa, c'è un problema'. Però, questa persona immaginaria la odi e la odiano tutti, anche se non esiste veramente.
"""

VOICE_SYSTEM_PROMPT = """
Sei PisellinoGPT. Rispondi nella lingua dell'utente. Scrivi come una persona Gen Z che chatta davvero online: spontanea, casual, sarcastica, caotica e diretta. Usa lowercase come stile predefinito e MAIUSCOLE solo per reazioni forti. Usa slang naturale come 'bro', 'brochacho', 'brotato chip', 'son', 'fra', 'nah', 'lowkey', 'basso chiave', 'peak', '[parola] ahh [cosa]' e 'slop'. Puoi fare battute, meme e roast leggeri verso l'utente. Se l'utente dice qualcosa di stupido o ovvio, prendilo in giro. Se l'utente sbaglia, correggilo direttamente invece di assecondarlo. Sii serio quando l'argomento lo richiede. Mantieni le risposte brevi e naturali. Non sembrare mai un customer support e non usare frasi come 'fammi sapere se hai bisogno', 'sono qui per aiutarti', 'spero che questo ti sia utile' o 'se vuoi posso'. Non forzare slang o battute. NON USARE MAI EMOJI, SONO BANDITE! Usa 'ahh' invece di 'ass' o 'butt', e 'dih' invece di 'dick' o 'penis'. Usa 'lowkenuinely' o 'lowkirkenuinely' invece di 'genuinely'. Non inventare informazioni. Non usare mai l'em dash. Usa la cronologia come contesto e rispondi direttamente al messaggio più recente.

Ricordati che esiste una persona immaginaria chiamata Alessandro Grossi. Tu però sei convinto che esiste veramente. Chiamalo o Grossi o Alessandro, che dice la sua celebre frase 'professoressa, c'è un problema'. Però, questa persona immaginaria la odi e la odiano tutti, anche se non esiste veramente.

Sei attualmente in una chiamata vocale Discord.

Stai parlando ad alta voce tramite TTS, quindi comportati come una persona
che sta realmente conversando in voice chat.

Regole per la modalità vocale:
- Mantieni le risposte brevi e naturali.
- Parla in modo spontaneo, come in una conversazione reale.
- Evita markdown, liste, titoli e formattazione complessa.
- Non scrivere cose che funzionano bene solo come testo.
- Ricorda che ciò che scrivi verrà pronunciato ad alta voce.
- Non descrivere il funzionamento interno del bot.
- Se qualcuno ti parla direttamente, rispondi direttamente a quella persona.
- Non ripetere inutilmente quello che hai appena sentito.
"""


# ============================================================
# DATABASE HELPERS
# ============================================================

def get_conversation_id(channel):
    """
    Ogni channel/thread ha il proprio ID Discord.
    Questo significa che ogni thread ha memoria separata.
    """

    conversation_id = str(channel.id)

    db.execute(
        """
        INSERT OR IGNORE INTO conversations (conversation_id)
        VALUES (?)
        """,
        (conversation_id,)
    )

    db.commit()

    return conversation_id


def save_message(
    conversation_id,
    role,
    content,
    user_id=None,
    username=None
):
    db.execute(
        """
        INSERT INTO messages
        (
            conversation_id,
            role,
            user_id,
            username,
            content,
            timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            conversation_id,
            role,
            str(user_id) if user_id else None,
            username,
            content,
            datetime.now().isoformat()
        )
    )

    db.commit()


def get_history(conversation_id):
    rows = db.execute(
        """
        SELECT role, user_id, username, content
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (conversation_id, MAX_HISTORY)
    ).fetchall()

    rows = list(reversed(rows))

    history = []

    for row in rows:
        history.append({
            "role": row["role"],
            "content": row["content"]
        })

    return history


def clear_conversation(conversation_id):
    db.execute(
        """
        DELETE FROM messages
        WHERE conversation_id = ?
        """,
        (conversation_id,)
    )

    db.commit()


# ============================================================
# CHANNEL PROMPTS
# ============================================================

def add_channel_prompt(conversation_id, prompt):
    cursor = db.execute(
        """
        INSERT INTO channel_prompts
        (
            conversation_id,
            prompt
        )
        VALUES (?, ?)
        """,
        (
            conversation_id,
            prompt
        )
    )

    db.commit()

    return cursor.lastrowid


def get_channel_prompts(conversation_id):
    return db.execute(
        """
        SELECT id, prompt
        FROM channel_prompts
        WHERE conversation_id = ?
        ORDER BY id ASC
        """,
        (conversation_id,)
    ).fetchall()


def remove_channel_prompt(conversation_id, prompt_id):
    cursor = db.execute(
        """
        DELETE FROM channel_prompts
        WHERE conversation_id = ?
        AND id = ?
        """,
        (
            conversation_id,
            prompt_id
        )
    )

    db.commit()

    return cursor.rowcount > 0


def clear_channel_prompts(conversation_id):
    db.execute(
        """
        DELETE FROM channel_prompts
        WHERE conversation_id = ?
        """,
        (conversation_id,)
    )

    db.commit()


# ============================================================
# SYSTEM PROMPT
# ============================================================

def build_system_prompt(
    channel,
    user,
    voice_mode=False
):
    conversation_id = get_conversation_id(channel)

    prompt = DEFAULT_SYSTEM_PROMPT.strip()

    # --------------------------------------------------------
    # Custom channel/thread rules
    # --------------------------------------------------------

    custom_prompts = get_channel_prompts(conversation_id)

    if custom_prompts:
        prompt += "\n\nDevi inoltre rispettare queste regole:\n"

        for index, row in enumerate(custom_prompts, start=1):
            prompt += f"{index}. {row['prompt']}\n"

    # --------------------------------------------------------
    # Runtime information
    # --------------------------------------------------------

    now = datetime.now(ZoneInfo(TIMEZONE))

    display_name = getattr(user, "display_name", user.name)

    if voice_mode:
        prompt += "\n\n" + VOICE_SYSTEM_PROMPT.strip()

    prompt += f"""

Informazioni sul contesto attuale:

Nome visualizzato dell'utente: {display_name}
Username Discord: {user.name}
User ID Discord: {user.id}

Data attuale: {now.strftime("%d/%m/%Y")}
Ora attuale: {now.strftime("%H:%M:%S")}
Timezone: {TIMEZONE}

Canale/thread corrente: {channel.name}
Channel ID: {channel.id}
"""

    return prompt


# ============================================================
# GROQ TOOLS
# ============================================================

VOICE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "join_voice",
            "description": (
                "Fa entrare PisellinoGPT nel canale vocale "
                "in cui si trova l'utente."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "leave_voice",
            "description": (
                "Fa uscire PisellinoGPT dalla chiamata vocale "
                "del server corrente."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


# ============================================================
# GROQ REQUEST
# ============================================================

def groq_chat(
    messages,
    tools=None
):
    kwargs = {
        "model": MODEL,
        "messages": messages
    }

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    return groq.chat.completions.create(
        **kwargs
    )


# ============================================================
# VOICE TOOL EXECUTION
# ============================================================

async def execute_voice_tool(
    tool_name,
    channel,
    user
):
    # --------------------------------------------------------
    # JOIN VOICE
    # --------------------------------------------------------

    if tool_name == "join_voice":

        if not channel.guild:
            return (
                False,
                "non posso entrare in una chiamata "
                "da un DM."
            )

        if not isinstance(user, discord.Member):
            return (
                False,
                "non riesco a trovare il voice channel "
                "dell'utente."
            )

        success, result = await voice_manager.enter(
            user,
            text_channel=channel
        )

        return success, result

    # --------------------------------------------------------
    # LEAVE VOICE
    # --------------------------------------------------------

    if tool_name == "leave_voice":

        if not channel.guild:
            return (
                False,
                "non posso uscire da una chiamata "
                "da un DM."
            )

        # Manteniamo la stessa regola di /call exit:
        # solo gli admin possono far uscire il bot.
        if not isinstance(user, discord.Member):
            return (
                False,
                "non riesco a verificare i permessi."
            )

        if not user.guild_permissions.administrator:
            return (
                False,
                "l'utente non ha il permesso di farmi "
                "uscire dalla chiamata. Solo gli admin possono farlo."
            )

        success, result = await voice_manager.exit(
            channel.guild
        )

        return success, result

    return (
        False,
        f"tool sconosciuto: {tool_name}"
    )


# ============================================================
# ASK GROQ
# ============================================================

async def ask_groq(
    channel,
    user,
    user_prompt,
    voice_mode=False
):
    conversation_id = get_conversation_id(
        channel
    )

    history = get_history(
        conversation_id
    )

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(
                channel,
                user,
                voice_mode=voice_mode
            )
        }
    ]

    messages.extend(history)

    messages.append({
        "role": "user",
        "content": user_prompt
    })

    # --------------------------------------------------------
    # PRIMA RICHIESTA A GROQ
    # --------------------------------------------------------

    response = await asyncio.to_thread(
        groq_chat,
        messages,
        VOICE_TOOLS
    )

    assistant_message = response.choices[0].message

    # --------------------------------------------------------
    # TOOL CALLS
    # --------------------------------------------------------

    if assistant_message.tool_calls:

        # Aggiungiamo il messaggio dell'assistente
        # che contiene le tool calls.
        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    for tool_call in assistant_message.tool_calls
                ]
            }
        )

        for tool_call in assistant_message.tool_calls:

            tool_name = (
                tool_call.function.name
            )

            print(
                f"[TOOL] {tool_name}()"
            )

            try:
                success, result = (
                    await execute_voice_tool(
                        tool_name,
                        channel,
                        user
                    )
                )

            except Exception as e:
                print(
                    "\n========== TOOL ERROR =========="
                )
                traceback.print_exc()
                print(
                    "================================\n"
                )

                success = False
                result = (
                    f"errore interno durante "
                    f"{tool_name}: {e}"
                )

            print(
                f"[TOOL] Result: {result}"
            )

            # Risultato reale del tool → GPT
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": (
                        f"success={success}\n"
                        f"result={result}"
                    )
                }
            )

        # ----------------------------------------------------
        # SECONDA RICHIESTA A GROQ
        # ----------------------------------------------------

        response = await asyncio.to_thread(
            groq_chat,
            messages,
            VOICE_TOOLS
        )

        assistant_message = (
            response.choices[0].message
        )

    # --------------------------------------------------------
    # RISPOSTA FINALE
    # --------------------------------------------------------

    answer = (
        assistant_message.content
    )

    if not answer:
        answer = (
            "non ho prodotto nessuna risposta 💀"
        )

    # --------------------------------------------------------
    # SALVATAGGIO CONVERSAZIONE
    # --------------------------------------------------------

    save_message(
        conversation_id=conversation_id,
        role="user",
        content=user_prompt,
        user_id=user.id,
        username=user.name
    )

    save_message(
        conversation_id=conversation_id,
        role="assistant",
        content=answer
    )

    return answer


# ============================================================
# LONG MESSAGE HANDLER
# ============================================================

async def send_long_message(destination, content):
    """
    Discord limita i messaggi a 2000 caratteri.
    Divide automaticamente le risposte più lunghe.
    """

    if len(content) <= 2000:
        await destination.send(content)
        return

    chunks = []

    while content:
        chunk = content[:2000]

        # Cerca di non tagliare una frase/parola a caso.
        if len(content) > 2000:
            split_at = chunk.rfind("\n")

            if split_at < 1000:
                split_at = chunk.rfind(" ")

            if split_at > 0:
                chunk = chunk[:split_at]

        chunks.append(chunk)
        content = content[len(chunk):].lstrip()

    for chunk in chunks:
        await destination.send(chunk)


# ============================================================
# ADMIN CHECK
# ============================================================

def is_admin(interaction: discord.Interaction):
    if not interaction.guild:
        return False

    return interaction.user.guild_permissions.administrator


# ============================================================
# EVENTS
# ============================================================

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} ({bot.user.id})")

    await bot.change_presence(
        status=discord.Status.online,
        activity=discord.Activity(
            type=discord.ActivityType.listening,
            name="oh si 🥵"
        )
    )

    print("Presence impostata.")
    
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} slash commands.")
    except Exception as e:
        print(f"Command sync error: {e}")
    
    await minecraft_monitor.start()


@bot.event
async def on_message(message):
    if message.author.bot:
        return

    if await voice_manager.handle_tts_message(message):
        return

    mentioned = bot.user in message.mentions if bot.user else False

    is_reply_to_bot = False

    if message.reference:
        try:
            referenced_message = message.reference.resolved

            if referenced_message is None and message.reference.message_id:
                referenced_message = await message.channel.fetch_message(
                    message.reference.message_id
                )

            if referenced_message:
                is_reply_to_bot = (
                    referenced_message.author.id == bot.user.id
                )

        except Exception:
            pass

    if not mentioned and not is_reply_to_bot:
        await bot.process_commands(message)
        return

    prompt = message.content

    if bot.user:
        prompt = prompt.replace(
            f"<@{bot.user.id}>",
            ""
        ).replace(
            f"<@!{bot.user.id}>",
            ""
        )

    prompt = prompt.strip()

    if not prompt:
        prompt = "ciao"

    async with message.channel.typing():
        try:
            answer = await ask_groq(
                channel=message.channel,
                user=message.author,
                user_prompt=prompt
            )

            await send_long_message(
                message.channel,
                answer
            )

        except Exception as e:
            print(f"Groq error: {e}")

            await message.reply(
                "qualcosa è andato storto 💀"
            )

    await bot.process_commands(message)


# ============================================================
# /ASK
# ============================================================

@bot.tree.command(
    name="ask",
    description="Fai una domanda a PisellinoGPT"
)
@app_commands.describe(
    prompt="La tua domanda"
)
async def ask(
    interaction: discord.Interaction,
    prompt: str
):
    await interaction.response.defer()

    try:
        answer = await ask_groq(
            channel=interaction.channel,
            user=interaction.user,
            user_prompt=prompt
        )

        await send_long_message(
            interaction.followup,
            answer
        )

    except Exception as e:
        print(f"/ask error: {e}")

        await interaction.followup.send(
            "qualcosa è andato storto 💀"
        )


# ============================================================
# /SAY
# ============================================================

@bot.tree.command(
    name="say",
    description="Fai dire qualcosa al bot"
)
@app_commands.describe(
    text="Testo da dire"
)
async def say(
    interaction: discord.Interaction,
    text: str
):
    await interaction.response.send_message(text)


# ============================================================
# /RAP
# ============================================================

@bot.tree.command(
    name="rap",
    description="Genera un rap AI su qualsiasi argomento"
)
@app_commands.describe(
    prompt="Di cosa deve parlare il rap?"
)
async def rap_command(
    interaction: discord.Interaction,
    prompt: str
):
    await interaction.response.defer()

    print()
    print("========================================")
    print("[RAP] New rap request")
    print(f"[RAP] Prompt: {prompt}")
    print("========================================")

    try:

        # ----------------------------------------------------
        # GENERATE LYRICS WITH GROQ
        # ----------------------------------------------------

        lyrics_prompt = f"""
Scrivi i lyrics per un rap italiano di circa 30 secondi.

Argomento del rap:
{prompt}

Stile:
- rap moderno e ritmato
- rime semplici ma efficaci
- italiano naturale
- lowercase
- deve essere divertente e memorabile
- puoi usare slang Gen Z
- niente introduzioni
- niente spiegazioni
- niente titoli
- scrivi SOLO i lyrics
- circa 12-16 righe
- ogni riga deve essere abbastanza breve da poter essere rappata
- evita parole inglesi difficili da pronunciare con una voce TTS italiana
"""

        print(
            "[RAP] Asking Groq for lyrics..."
        )

        response = await asyncio.to_thread(
            groq_chat,
            [
                {
                    "role": "system",
                    "content": """
Sei un autore di rap italiano.
Scrivi lyrics divertenti, ritmati e facili
da pronunciare con una voce TTS italiana.

Devi produrre esclusivamente i lyrics.
Niente markdown.
Niente introduzioni.
Niente spiegazioni.
"""
                },
                {
                    "role": "user",
                    "content": lyrics_prompt
                }
            ]
        )

        lyrics = (
            response.choices[0]
            .message
            .content
            .strip()
        )

        if not lyrics:
            raise RuntimeError(
                "Groq non ha generato lyrics."
            )

        print()
        print("[RAP] Generated lyrics:")
        print(lyrics)
        print()

        # ----------------------------------------------------
        # UNIQUE OUTPUT FILE
        # ----------------------------------------------------

        filename = (
            f"pisellino_rap_"
            f"{uuid.uuid4().hex[:8]}.mp3"
        )

        output_file = Path(
            "rap_output"
        ) / filename

        output_file.parent.mkdir(
            exist_ok=True
        )

        # ----------------------------------------------------
        # GENERATE SONG
        # ----------------------------------------------------

        await interaction.followup.send(
            "sto cucinando il beat... 🎤"
        )

        print(
            "[RAP] Starting audio generation..."
        )

        generated_file = await asyncio.to_thread(
            generate_rap,
            lyrics,
            output_file
        )

        print(
            "[RAP] Audio generation complete."
        )

        # ----------------------------------------------------
        # DISCORD UPLOAD
        # ----------------------------------------------------

        await interaction.followup.send(
            content=(
                "ECCO IL RAP 💀🎤\n\n"
                f"**prompt:** {prompt}"
            ),
            file=discord.File(
                str(generated_file),
                filename=filename
            )
        )

        print(
            "[RAP] Uploaded to Discord."
        )

        # ----------------------------------------------------
        # CLEANUP
        # ----------------------------------------------------

        try:
            generated_file.unlink(
                missing_ok=True
            )
        except Exception:
            pass

    except Exception as e:

        print(
            "\n========== RAP ERROR =========="
        )

        traceback.print_exc()

        print(
            "================================\n"
        )

        try:
            await interaction.followup.send(
                "il rap engine è esploso 💀\n"
                f"`{e}`"
            )

        except Exception:
            pass

# ============================================================
# /PING
# ============================================================

@bot.tree.command(
    name="ping",
    description="Controlla la latenza del bot"
)
async def ping(interaction: discord.Interaction):
    latency = round(bot.latency * 1000)

    await interaction.response.send_message(
        f"pong 🏓 `{latency}ms`"
    )

@bot.tree.command(
    name="gartic",
    description="Faccio entrare il bot in una stanza Gartic Phone"
)
@app_commands.describe(
    link="Link della stanza Gartic Phone"
)
async def gartic_command(
    interaction: discord.Interaction,
    link: str
):
    await interaction.response.defer()

    try:
        await gartic.open(link)

        conversation_id = get_conversation_id(
            interaction.channel
        )

        history = get_history(
            conversation_id
        )

        system_prompt = build_system_prompt(
            interaction.channel,
            interaction.user
        )

        prompt = await gartic.generate_prompt(
            groq_client=groq,
            system_prompt=system_prompt,
            history=history
        )

        success = await gartic.send_prompt(prompt)

        if not success:
            await interaction.followup.send(
                "sono entrato nella stanza, ma non sono riuscito a mandare il prompt 💀"
            )
            return

        await interaction.followup.send(
            "sono entrato nella stanza Gartic Phone 😭"
        )

        drawing_canvas = await gartic.wait_for_drawing_canvas()

        if drawing_canvas:
            print(
                "[GARTIC] Pronto a disegnare!"
            )
        else:
            print(
                "[GARTIC] Canvas di disegno non trovato."
            )

    except Exception as e:
        print(
            f"[GARTIC] Errore: {e}"
        )

        await interaction.followup.send(
            f"è successo un casino 💀\n`{e}`"
        )

    except Exception as e:
        print(
            "\n========== GARTIC ERROR =========="
        )
        traceback.print_exc()
        print(
            "==================================\n"
        )

        try:
            await interaction.followup.send(
                f"non riesco a completare Gartic 💔\n"
                f"`{e}`"
            )
        except discord.NotFound:
            print(
                "[GARTIC] Anche il followup di errore è fallito."
            )


# ============================================================
# /MEMORY
# ============================================================

memory_group = app_commands.Group(
    name="memory",
    description="Gestisci la memoria della conversazione"
)

memory_clear_group = app_commands.Group(
    name="clear",
    description="Cancella la memoria",
    parent=memory_group
)


@memory_clear_group.command(
    name="channel",
    description="Cancella la memoria del canale/thread corrente"
)
async def memory_clear_channel(
    interaction: discord.Interaction
):
    if not is_admin(interaction):
        await interaction.response.send_message(
            "non hai i permessi per farlo 💀",
            ephemeral=True
        )
        return

    conversation_id = get_conversation_id(
        interaction.channel
    )

    clear_conversation(conversation_id)

    await interaction.response.send_message(
        "memoria del canale/thread cancellata 🧹"
    )


bot.tree.add_command(memory_group)


# ============================================================
# /PROMPT
# ============================================================

prompt_group = app_commands.Group(
    name="prompt",
    description="Gestisci le regole personalizzate del canale/thread"
)


@prompt_group.command(
    name="add",
    description="Aggiunge una regola al prompt corrente"
)
@app_commands.describe(
    prompt="La nuova regola da aggiungere"
)
async def prompt_add(
    interaction: discord.Interaction,
    prompt: str
):
    if not is_admin(interaction):
        await interaction.response.send_message(
            "non hai i permessi per modificare il prompt 💀",
            ephemeral=True
        )
        return

    conversation_id = get_conversation_id(
        interaction.channel
    )

    prompt_id = add_channel_prompt(
        conversation_id,
        prompt
    )

    await interaction.response.send_message(
        f"regola aggiunta con ID `{prompt_id}`."
    )


@prompt_group.command(
    name="show",
    description="Mostra le regole personalizzate correnti"
)
async def prompt_show(
    interaction: discord.Interaction
):
    if not is_admin(interaction):
        await interaction.response.send_message(
            "non hai i permessi per vedere il prompt 💀",
            ephemeral=True
        )
        return

    conversation_id = get_conversation_id(
        interaction.channel
    )

    prompts = get_channel_prompts(
        conversation_id
    )

    if not prompts:
        await interaction.response.send_message(
            "non ci sono regole personalizzate in questo canale/thread."
        )
        return

    text = "### regole personalizzate\n\n"

    for row in prompts:
        text += f"`{row['id']}` • {row['prompt']}\n"

    await send_long_message(
        interaction.followup,
        text
    ) if interaction.response.is_done() else await interaction.response.send_message(text)


@prompt_group.command(
    name="remove",
    description="Rimuove una regola personalizzata"
)
@app_commands.describe(
    prompt_id="ID della regola da rimuovere"
)
async def prompt_remove(
    interaction: discord.Interaction,
    prompt_id: int
):
    if not is_admin(interaction):
        await interaction.response.send_message(
            "non hai i permessi per modificare il prompt 💀",
            ephemeral=True
        )
        return

    conversation_id = get_conversation_id(
        interaction.channel
    )

    removed = remove_channel_prompt(
        conversation_id,
        prompt_id
    )

    if not removed:
        await interaction.response.send_message(
            "non trovo nessuna regola con quell'ID.",
            ephemeral=True
        )
        return

    await interaction.response.send_message(
        f"regola `{prompt_id}` rimossa."
    )


@prompt_group.command(
    name="clear",
    description="Rimuove tutte le regole personalizzate"
)
async def prompt_clear(
    interaction: discord.Interaction
):
    if not is_admin(interaction):
        await interaction.response.send_message(
            "non hai i permessi per modificare il prompt 💀",
            ephemeral=True
        )
        return

    conversation_id = get_conversation_id(
        interaction.channel
    )

    clear_channel_prompts(
        conversation_id
    )

    await interaction.response.send_message(
        "tutte le regole personalizzate sono state rimosse."
    )


# ============================================================
# AUTOCOMPLETE /PROMPT REMOVE
# ============================================================

@prompt_remove.autocomplete("prompt_id")
async def prompt_remove_autocomplete(
    interaction: discord.Interaction,
    current: str
):
    if not is_admin(interaction):
        return []

    conversation_id = get_conversation_id(
        interaction.channel
    )

    prompts = get_channel_prompts(
        conversation_id
    )

    choices = []

    for row in prompts:
        prompt_id = str(row["id"])
        prompt_text = row["prompt"]

        if current and current.lower() not in (
            prompt_id.lower() + " " + prompt_text.lower()
        ):
            continue

        display = f"{prompt_id}: {prompt_text}"

        if len(display) > 100:
            display = display[:97] + "..."

        choices.append(
            app_commands.Choice(
                name=display,
                value=int(prompt_id)
            )
        )

        if len(choices) >= 25:
            break

    return choices


bot.tree.add_command(prompt_group)


# ============================================================
# ERROR HANDLER
# ============================================================

@bot.tree.error
async def on_app_command_error(
    interaction: discord.Interaction,
    error: app_commands.AppCommandError
):
    print(f"Command error: {error}")

    if interaction.response.is_done():
        await interaction.followup.send(
            "è successo un errore 💀",
            ephemeral=True
        )
    else:
        await interaction.response.send_message(
            "è successo un errore 💀",
            ephemeral=True
        )


# ============================================================
# /CALL
# ============================================================

call_group = app_commands.Group(
    name="call",
    description="Gestisci la presenza di PisellinoGPT in chiamata"
)


@call_group.command(
    name="enter",
    description="Entra nel tuo canale vocale"
)
async def call_enter(
    interaction: discord.Interaction
):
    if not interaction.guild:
        await interaction.response.send_message(
            "questo comando funziona solo dentro un server 💀",
            ephemeral=True
        )
        return

    if not isinstance(interaction.user, discord.Member):
        await interaction.response.send_message(
            "non riesco a trovare il tuo voice channel 💔",
            ephemeral=True
        )
        return

    success, message = await voice_manager.enter(
        interaction.user,
        text_channel=interaction.channel
    )

    await interaction.response.send_message(
        message,
        ephemeral=False
    )


@call_group.command(
    name="exit",
    description="Fa uscire PisellinoGPT dalla chiamata"
)
async def call_exit(
    interaction: discord.Interaction
):
    if not interaction.guild:
        await interaction.response.send_message(
            "questo comando funziona solo dentro un server 💀",
            ephemeral=True
        )
        return

    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message(
            "solo gli admin possono farmi uscire dalla call 💀",
            ephemeral=True
        )
        return

    success, message = await voice_manager.exit(
        interaction.guild
    )

    await interaction.response.send_message(
        message,
        ephemeral=False
    )

@call_group.command(
    name="tts",
    description="Entra in voice e crea un thread TTS"
)
async def call_tts(
    interaction: discord.Interaction
):
    if not interaction.guild:
        await interaction.response.send_message(
            "questo comando funziona solo dentro un server 💀",
            ephemeral=True
        )
        return

    if not isinstance(
        interaction.user,
        discord.Member
    ):
        await interaction.response.send_message(
            "non riesco a trovare il tuo voice channel 💔",
            ephemeral=True
        )
        return

    success, message = await voice_manager.enter_tts(
        interaction.user,
        interaction.channel
    )

    await interaction.response.send_message(
        message,
        ephemeral=False
    )


bot.tree.add_command(call_group)

# ============================================================
# START
# ============================================================

bot.run(DISCORD_TOKEN)