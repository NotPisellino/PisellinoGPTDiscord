import asyncio
import io
import os
import tempfile
import time
import traceback
import wave

import discord
from discord.ext import voice_recv
from gtts import gTTS


VOICE_TIMEOUT = 10 * 60

SPEECH_END_DELAY = 0.8

WHISPER_MODEL = "whisper-large-v3-turbo"

TTS_LANGUAGE = "it"


class SpeechSink(voice_recv.AudioSink):
    def __init__(
        self,
        groq_client,
        loop,
        on_transcript=None,
        on_leave=None
    ):
        super().__init__()

        self.groq = groq_client
        self.loop = loop
        self.on_transcript = on_transcript
        self.on_leave = on_leave

        self.buffers: dict[int, bytearray] = {}
        self.users: dict[int, object] = {}
        self.last_audio: dict[int, float] = {}

        self.processing: set[int] = set()

        self.paused = False
        self.running = True

        self.monitor_task = asyncio.create_task(
            self._monitor_speakers()
        )

    def wants_opus(self) -> bool:
        return False

    def write(self, user, data):
        if not user:
            return

        if getattr(user, "bot", False):
            return

        if self.paused:
            return

        if not data.pcm:
            return

        user_id = user.id

        if user_id not in self.buffers:
            self.buffers[user_id] = bytearray()

        self.buffers[user_id].extend(data.pcm)

        self.users[user_id] = user
        self.last_audio[user_id] = time.monotonic()

    async def _monitor_speakers(self):
        while self.running:
            try:
                now = time.monotonic()

                for user_id in list(self.buffers.keys()):
                    if user_id in self.processing:
                        continue

                    last_audio = self.last_audio.get(
                        user_id
                    )

                    if last_audio is None:
                        continue

                    if (
                        now - last_audio
                        >= SPEECH_END_DELAY
                    ):
                        audio_data = bytes(
                            self.buffers.pop(
                                user_id,
                                bytearray()
                            )
                        )

                        self.last_audio.pop(
                            user_id,
                            None
                        )

                        if not audio_data:
                            continue

                        self.processing.add(
                            user_id
                        )

                        asyncio.create_task(
                            self._transcribe(
                                user_id,
                                self.users.get(
                                    user_id
                                ),
                                audio_data
                            )
                        )

                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break

            except Exception:
                print(
                    "\n========== SPEECH MONITOR ERROR =========="
                )
                traceback.print_exc()
                print(
                    "===========================================\n"
                )

                await asyncio.sleep(1)

    def _is_leave_command(
        self,
        text: str
    ) -> bool:
        text = text.lower().strip()

        phrases = [
            "esci dalla chiamata",
            "esci dalla call",
            "escitene dalla chiamata",
            "escitene dalla call",
            "vattene dalla chiamata",
            "vattene dalla call",
            "vai via dalla chiamata",
            "vai via dalla call",
            "lascia la chiamata",
            "lascia la call",
            "esci dalla chat vocale",
            "esci dalla chat",
            "lascia la chat vocale",
            "lascia la chat"
        ]

        return any(
            phrase in text
            for phrase in phrases
        )

    async def _transcribe(
        self,
        user_id,
        user,
        pcm_data
    ):
        try:
            wav_data = self._make_wav(
                pcm_data
            )

            transcription = await asyncio.to_thread(
                self._groq_transcribe,
                wav_data
            )

            text = transcription.strip()

            if not text:
                return

            username = getattr(
                user,
                "display_name",
                getattr(
                    user,
                    "name",
                    str(user_id)
                )
            )

            print(
                f'\n[VOICE] {username}: "{text}"'
            )

            if self._is_leave_command(
                text
            ):
                print(
                    f"[VOICE] Leave command detected "
                    f"from {username}"
                )

                guild = getattr(
                    user,
                    "guild",
                    None
                )

                if guild and self.on_leave:
                    asyncio.create_task(
                        self.on_leave(guild)
                    )

                return

            if self.on_transcript:
                asyncio.create_task(
                    self._send_to_gpt(
                        user,
                        text
                    )
                )

        except Exception:
            print(
                "\n========== WHISPER ERROR =========="
            )
            traceback.print_exc()
            print(
                "===================================\n"
            )

        finally:
            self.processing.discard(
                user_id
            )

    async def _send_to_gpt(
        self,
        user,
        text
    ):
        try:
            response = await self.on_transcript(
                user,
                text
            )

            if response:
                print(
                    f'\n[VOICE → GPT] {response}'
                )

        except Exception:
            print(
                "\n========== VOICE GPT ERROR =========="
            )
            traceback.print_exc()
            print(
                "======================================\n"
            )

    def _groq_transcribe(
        self,
        wav_data: bytes
    ):
        file_object = io.BytesIO(
            wav_data
        )

        file_object.name = (
            "discord_voice.wav"
        )

        response = (
            self.groq.audio.transcriptions.create(
                file=file_object,
                model=WHISPER_MODEL,
                language="it",
                prompt=(
                    "La persona sta parlando "
                    "esclusivamente in italiano. "
                    "Trascrivi fedelmente le parole "
                    "pronunciate in italiano. "
                    "Non tradurre e non interpretare "
                    "in altre lingue."
                ),
                response_format="text",
                temperature=0.0
            )
        )

        return response

    @staticmethod
    def _make_wav(
        pcm_data: bytes
    ) -> bytes:
        output = io.BytesIO()

        with wave.open(
            output,
            "wb"
        ) as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(48000)

            wav.writeframes(
                pcm_data
            )

        return output.getvalue()

    def cleanup(self):
        self.running = False

        if (
            self.monitor_task
            and not self.monitor_task.done()
        ):
            self.monitor_task.cancel()

        self.buffers.clear()
        self.users.clear()
        self.last_audio.clear()
        self.processing.clear()


class VoiceManager:
    def __init__(
        self,
        groq_client,
        on_transcript=None
    ):
        self.groq = groq_client
        self.on_transcript = on_transcript

        self.timeout_tasks: dict[
            int,
            asyncio.Task
        ] = {}

        self.sinks: dict[
            int,
            SpeechSink
        ] = {}

        self.text_channels: dict[
            int,
            discord.abc.Messageable
        ] = {}

        self.tts_locks: dict[
            int,
            asyncio.Lock
        ] = {}

        self.tts_threads: dict[
            int,
            discord.Thread
        ] = {}

        self.tts_mode: set[int] = set()

    async def _handle_voice_leave(
        self,
        guild
    ):
        print(
            f"[VOICE] Voice leave command detected "
            f"in {guild.name}"
        )

        await self.exit(guild)

    async def enter(
        self,
        member: discord.Member,
        text_channel=None
    ):
        if (
            not member.voice
            or not member.voice.channel
        ):
            return False, (
                "devi essere in un canale vocale."
            )

        voice_channel = member.voice.channel
        guild = member.guild
        guild_id = guild.id

        self.tts_mode.discard(
            guild_id
        )

        if text_channel:
            self.text_channels[
                guild_id
            ] = text_channel

        permissions = (
            voice_channel.permissions_for(
                guild.me
            )
        )

        if not permissions.connect:
            return False, (
                f"non ho il permesso `Connect` "
                f"in **{voice_channel.name}** 💔"
            )

        if not permissions.speak:
            return False, (
                f"non ho il permesso `Speak` "
                f"in **{voice_channel.name}** 💔"
            )

        try:
            if guild.voice_client:
                voice_client = (
                    guild.voice_client
                )

                if (
                    voice_client.channel.id
                    != voice_channel.id
                ):
                    await voice_client.move_to(
                        voice_channel
                    )

                if isinstance(
                    voice_client,
                    voice_recv.VoiceRecvClient
                ):
                    self._start_listening(
                        guild,
                        voice_client
                    )

                self.reset_timeout(
                    guild
                )

                return True, (
                    f"sono entrato in "
                    f"**{voice_channel.name}** 😭"
                )

            voice_client = (
                await voice_channel.connect(
                    cls=voice_recv.VoiceRecvClient,
                    timeout=30.0,
                    reconnect=True
                )
            )

            self._start_listening(
                guild,
                voice_client
            )

            self.reset_timeout(
                guild
            )

            print(
                f"[VOICE] Connected to "
                f"{guild.name} / "
                f"{voice_channel.name}"
            )

            return True, (
                f"sono entrato in "
                f"**{voice_channel.name}** 😭"
            )

        except asyncio.TimeoutError:
            print(
                "\n========== VOICE TIMEOUT =========="
            )
            traceback.print_exc()
            print(
                "===================================\n"
            )

            return False, (
                "discord non ha completato "
                "la connessione voice entro "
                "30 secondi 💔"
            )

        except Exception:
            print(
                "\n========== VOICE CONNECTION ERROR =========="
            )
            traceback.print_exc()
            print(
                "============================================\n"
            )

            return False, (
                "non riesco a entrare nella "
                "chiamata 💔\n"
                "guarda il terminale per "
                "l'errore vero."
            )

    async def enter_tts(
        self,
        member: discord.Member,
        text_channel
    ):
        if (
            not member.voice
            or not member.voice.channel
        ):
            return False, (
                "devi essere in un canale vocale."
            )

        if not isinstance(
            text_channel,
            discord.TextChannel
        ):
            return False, (
                "usa `/call tts` da un "
                "canale testuale normale 💔"
            )

        voice_channel = member.voice.channel
        guild = member.guild
        guild_id = guild.id

        permissions = (
            voice_channel.permissions_for(
                guild.me
            )
        )

        if not permissions.connect:
            return False, (
                f"non ho il permesso `Connect` "
                f"in **{voice_channel.name}** 💔"
            )

        if not permissions.speak:
            return False, (
                f"non ho il permesso `Speak` "
                f"in **{voice_channel.name}** 💔"
            )

        try:
            if guild.voice_client:
                voice_client = (
                    guild.voice_client
                )

                if (
                    voice_client.channel.id
                    != voice_channel.id
                ):
                    await voice_client.move_to(
                        voice_channel
                    )

            else:
                voice_client = (
                    await voice_channel.connect(
                        cls=voice_recv.VoiceRecvClient,
                        timeout=30.0,
                        reconnect=True
                    )
                )

            # TTS mode NON usa Whisper.
            self._stop_listening(
                guild
            )

            self.text_channels[
                guild_id
            ] = text_channel

            self.tts_mode.add(
                guild_id
            )

            old_thread = self.tts_threads.get(
                guild_id
            )

            if (
                old_thread
                and not old_thread.archived
                and not old_thread.locked
            ):
                self.reset_timeout(
                    guild
                )

                return True, (
                    f"modalità TTS già attiva "
                    f"in {old_thread.mention} 🔊"
                )

            thread = await text_channel.create_thread(
                name="🔊 tts",
                type=discord.ChannelType.public_thread
            )

            self.tts_threads[
                guild_id
            ] = thread

            self.reset_timeout(
                guild
            )

            print(
                f"[VOICE TTS] Connected to "
                f"{guild.name} / "
                f"{voice_channel.name}"
            )

            print(
                f"[VOICE TTS] Thread created: "
                f"{thread.id}"
            )

            return True, (
                f"sono entrato in "
                f"**{voice_channel.name}** 🔊\n"
                f"scrivi quello che vuoi nel "
                f"{thread.mention} e lo leggerò."
            )

        except asyncio.TimeoutError:
            return False, (
                "discord non ha completato "
                "la connessione voice entro "
                "30 secondi 💔"
            )

        except Exception:
            print(
                "\n========== TTS ENTER ERROR =========="
            )
            traceback.print_exc()
            print(
                "=====================================\n"
            )

            return False, (
                "non riesco ad avviare "
                "la modalità TTS 💔"
            )

    async def handle_tts_message(
        self,
        message: discord.Message
    ):
        if message.author.bot:
            return False

        guild = message.guild

        if not guild:
            return False

        guild_id = guild.id

        if guild_id not in self.tts_mode:
            return False

        thread = self.tts_threads.get(
            guild_id
        )

        if not thread:
            return False

        if message.channel.id != thread.id:
            return False

        text = message.content.strip()

        if not text:
            return True

        print(
            f'\n[VOICE TTS] '
            f'{message.author.display_name}: '
            f'"{text}"'
        )

        await self.speak(
            guild,
            text
        )

        return True

    def _start_listening(
        self,
        guild,
        voice_client
    ):
        guild_id = guild.id

        if guild_id in self.sinks:
            return

        loop = (
            asyncio.get_running_loop()
        )

        sink = SpeechSink(
            groq_client=self.groq,
            loop=loop,
            on_transcript=self._handle_transcript,
            on_leave=self._handle_voice_leave
        )

        self.sinks[
            guild_id
        ] = sink

        voice_client.listen(
            sink
        )

        print(
            f"[VOICE] Listening started "
            f"in {guild.name}"
        )

    async def _handle_transcript(
        self,
        user,
        text
    ):
        if not self.on_transcript:
            return None

        text_channel = (
            self.text_channels.get(
                user.guild.id
            )
        )

        if not text_channel:
            print(
                "[VOICE] Nessun text channel "
                "associato alla voice call."
            )
            return None

        answer = await self.on_transcript(
            text_channel,
            user,
            text
        )

        if answer:
            await self.speak(
                user.guild,
                answer
            )

        return answer

    async def speak(
        self,
        guild,
        text: str
    ):
        if not text:
            return

        voice_client = guild.voice_client

        if not voice_client:
            print(
                "[TTS] Nessun voice client."
            )
            return

        if not voice_client.is_connected():
            print(
                "[TTS] Voice client non connesso."
            )
            return

        guild_id = guild.id

        if guild_id not in self.tts_locks:
            self.tts_locks[
                guild_id
            ] = asyncio.Lock()

        async with self.tts_locks[
            guild_id
        ]:
            sink = self.sinks.get(
                guild_id
            )

            mp3_path = None

            try:
                print(
                    f'\n[TTS] Generating: "{text}"'
                )

                if sink:
                    sink.paused = True
                    sink.buffers.clear()
                    sink.last_audio.clear()

                # IMPORTANTISSIMO:
                # gTTS è bloccante, quindi viene
                # eseguito in un thread separato.
                mp3_path = await asyncio.to_thread(
                    self._generate_tts,
                    text
                )

                print(
                    f"[TTS] Playing in {guild.name}"
                )

                if not voice_client.is_connected():
                    return

                while voice_client.is_playing():
                    await asyncio.sleep(
                        0.1
                    )

                loop = (
                    asyncio.get_running_loop()
                )

                finished = loop.create_future()

                def after_playback(error):
                    if error:
                        print(
                            "\n========== TTS PLAYBACK ERROR =========="
                        )
                        print(error)
                        print(
                            "========================================\n"
                        )

                    if not finished.done():
                        loop.call_soon_threadsafe(
                            finished.set_result,
                            None
                        )

                source = discord.FFmpegPCMAudio(
                    mp3_path,
                    options="-vn"
                )

                voice_client.play(
                    source,
                    after=after_playback
                )

                await finished

                await asyncio.sleep(
                    0.15
                )

                if sink:
                    sink.buffers.clear()
                    sink.last_audio.clear()

                print(
                    "[TTS] Playback finished"
                )

            except Exception:
                print(
                    "\n========== TTS ERROR =========="
                )
                traceback.print_exc()
                print(
                    "===============================\n"
                )

            finally:
                if sink:
                    sink.paused = False
                    sink.buffers.clear()
                    sink.last_audio.clear()

                if (
                    mp3_path
                    and os.path.exists(mp3_path)
                ):
                    try:
                        os.remove(
                            mp3_path
                        )
                    except Exception:
                        pass

    @staticmethod
    def _generate_tts(
        text: str
    ) -> str:
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".mp3",
            delete=False
        )

        temp_file.close()

        try:
            tts = gTTS(
                text=text,
                lang=TTS_LANGUAGE,
                slow=False
            )

            tts.save(
                temp_file.name
            )

            return temp_file.name

        except Exception:
            try:
                if os.path.exists(
                    temp_file.name
                ):
                    os.remove(
                        temp_file.name
                    )
            except Exception:
                pass

            raise

    async def exit(
        self,
        guild
    ):
        voice_client = (
            guild.voice_client
        )

        if not voice_client:
            return False, (
                "non sono in nessuna chiamata."
            )

        self.cancel_timeout(
            guild.id
        )

        try:
            self._stop_listening(
                guild
            )

            self.tts_mode.discard(
                guild.id
            )

            self.tts_threads.pop(
                guild.id,
                None
            )

            self.text_channels.pop(
                guild.id,
                None
            )

            self.tts_locks.pop(
                guild.id,
                None
            )

            await voice_client.disconnect()

            print(
                f"[VOICE] Disconnected "
                f"from {guild.name}"
            )

            return True, (
                "me ne sono andato "
                "dalla chiamata 😭"
            )

        except Exception:
            print(
                "\n========== VOICE DISCONNECT ERROR =========="
            )
            traceback.print_exc()
            print(
                "============================================\n"
            )

            return False, (
                "non riesco a uscire "
                "dalla chiamata 💔"
            )

    def _stop_listening(
        self,
        guild
    ):
        sink = self.sinks.pop(
            guild.id,
            None
        )

        if sink:
            sink.cleanup()

        voice_client = (
            guild.voice_client
        )

        if isinstance(
            voice_client,
            voice_recv.VoiceRecvClient
        ):
            try:
                voice_client.stop_listening()
            except Exception:
                pass

    def reset_timeout(
        self,
        guild
    ):
        self.cancel_timeout(
            guild.id
        )

        self.timeout_tasks[
            guild.id
        ] = asyncio.create_task(
            self._timeout(guild)
        )

    def cancel_timeout(
        self,
        guild_id
    ):
        task = self.timeout_tasks.pop(
            guild_id,
            None
        )

        if (
            task
            and not task.done()
        ):
            task.cancel()

    async def _timeout(
        self,
        guild
    ):
        try:
            await asyncio.sleep(
                VOICE_TIMEOUT
            )

            voice_client = (
                guild.voice_client
            )

            if voice_client:
                print(
                    f"[VOICE] 10 minute "
                    f"timeout reached "
                    f"for {guild.name}"
                )

                self._stop_listening(
                    guild
                )

                self.tts_mode.discard(
                    guild.id
                )

                self.tts_threads.pop(
                    guild.id,
                    None
                )

                self.text_channels.pop(
                    guild.id,
                    None
                )

                self.tts_locks.pop(
                    guild.id,
                    None
                )

                await voice_client.disconnect()

        except asyncio.CancelledError:
            pass

        except Exception:
            print(
                "\n========== VOICE TIMEOUT ERROR =========="
            )
            traceback.print_exc()
            print(
                "=========================================\n"
            )

        finally:
            self.timeout_tasks.pop(
                guild.id,
                None
            )