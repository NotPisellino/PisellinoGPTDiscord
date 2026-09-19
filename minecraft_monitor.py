import asyncio
import os
from datetime import datetime

import discord
from mcstatus import JavaServer


class MinecraftMonitor:
    def __init__(self, bot):
        self.bot = bot

        self.host = "127.0.0.1"
        self.port = 25565

        self.channel_id = int(os.getenv("MC_STATUS_CHANNEL_ID", "0"))

        self.message = None
        self.server_was_online = False

        self.task = None

    async def start(self):
        if self.task is None:
            self.task = asyncio.create_task(self.monitor_loop())
            print("[MC] Minecraft monitor avviato.")

    async def get_status(self):
        def check():
            try:
                server = JavaServer.lookup(self.host)
                return server.status()
            except Exception:
                return None

        return await asyncio.to_thread(check)

    async def find_existing_message(self, channel):
        try:
            async for message in channel.history(limit=50):
                if (
                    message.author == self.bot.user
                    and message.embeds
                    and message.embeds[0].title == "PisellinoMC"
                ):
                    return message
        except Exception as e:
            print(f"[MC] Errore cercando il messaggio: {e}")

        return None

    def create_embed(self, status):
        online = status.players.online
        max_players = status.players.max

        version = status.version.name

        embed = discord.Embed(
            title="PisellinoMC",
            description=(
                "Il server è aperto!\n"
                "```mc.pisellino.dev```\n"
                f"**Giocatori**:\n"
                f"`{online}/{max_players}`\n\n"
                f"**Versione**:\n"
                f"`{version}`"
            ),
            color=16763525
        )

        embed.set_footer(text="oh si")

        return embed

    async def handle_online(self, channel, status):
        embed = self.create_embed(status)

        # Se non abbiamo ancora il messaggio, proviamo a recuperarlo
        if self.message is None:
            self.message = await self.find_existing_message(channel)

        # Se esiste già, aggiorniamolo
        if self.message is not None:
            try:
                await self.message.edit(embed=embed)
                return
            except discord.NotFound:
                self.message = None
            except discord.HTTPException as e:
                print(f"[MC] Errore aggiornando embed: {e}")
                return

        # Altrimenti creiamo un nuovo messaggio
        try:
            self.message = await channel.send(embed=embed)
            print("[MC] Server online, embed creato.")
        except discord.HTTPException as e:
            print(f"[MC] Errore creando embed: {e}")

    async def handle_offline(self):
        if self.message is None:
            return

        try:
            await self.message.delete()
            print("[MC] Server offline, embed cancellato.")
        except discord.NotFound:
            pass
        except discord.HTTPException as e:
            print(f"[MC] Errore cancellando embed: {e}")

        self.message = None

    async def monitor_loop(self):
        await self.bot.wait_until_ready()

        while not self.bot.is_closed():
            try:
                if not self.channel_id:
                    await asyncio.sleep(30)
                    continue

                channel = self.bot.get_channel(self.channel_id)

                if channel is None:
                    try:
                        channel = await self.bot.fetch_channel(self.channel_id)
                    except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                        await asyncio.sleep(30)
                        continue

                status = await self.get_status()

                if status is not None:
                    if not self.server_was_online:
                        print("[MC] 🟢 Server Minecraft ONLINE.")

                    await self.handle_online(channel, status)
                    self.server_was_online = True

                else:
                    if self.server_was_online:
                        print("[MC] 🔴 Server Minecraft OFFLINE.")

                    await self.handle_offline()
                    self.server_was_online = False

            except Exception:
                pass

            await asyncio.sleep(10)