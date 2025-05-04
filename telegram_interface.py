import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os       
import sys      
import traceback 

try:
    from chatbot import predict_class, get_response
    logging.info("Funciones 'predict_class' y 'get_response' importadas correctamente desde chatbot.py.")
    logging.info("La instancia global 'predictor' en chatbot.py debería haberse inicializado (modelo y datos cargados).")

except ImportError:
    logging.critical("¡ERROR CRÍTICO! No se pudo importar desde 'chatbot.py'.")
    logging.critical("Asegúrate de que 'chatbot.py' exista en el mismo directorio que este script.")
    sys.exit("Fallo al importar chatbot.py. Revisa el archivo y su ubicación.")
    
except FileNotFoundError as fnf_error:
     logging.critical(f"¡ERROR CRÍTICO! No se encontró un archivo necesario durante la inicialización de chatbot.py: {fnf_error}")
     logging.critical("Verifica las rutas a 'models/' y 'intents/' y sus contenidos dentro de chatbot.py.")
     logging.critical(traceback.format_exc())
     sys.exit("Fallo al cargar archivos del chatbot. Revisa los logs y chatbot.py.")
except Exception as e:
    logging.critical(f"¡ERROR CRÍTICO! Ocurrió un error al importar o inicializar 'chatbot.py': {e}")
    logging.critical(traceback.format_exc()) # Imprime el stack trace completo
    sys.exit("Fallo durante la inicialización del chatbot. Revisa los logs y chatbot.py.")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class TelegramBot:
    """Gestiona la interacción entre el chatbot y la API de Telegram."""
    def __init__(self, telegram_token: str):
        """Inicializa la aplicación de Telegram y registra los manejadores."""
        if not telegram_token:
            raise ValueError("El token de Telegram no puede estar vacío.")

        self.token = telegram_token
        logger.info("Creando instancia de Application de Telegram...")
        self.application = (
            Application.builder()
            .token(self.token)
            .read_timeout(30)
            .get_updates_read_timeout(30)
            .connect_timeout(30)
            .pool_timeout(30)
            .build()
        )
        logger.info("Application de Telegram creada.")
        self._register_handlers()

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Responde al comando /start."""
        user = update.effective_user
        logger.info(f"Comando /start recibido de {user.first_name} (ID: {user.id})")
        await update.message.reply_html(
            rf"¡Hola {user.mention_html()}! 👋 Soy tu <b>Asistente de Fútbol</b>. "
            rf"Pregúntame sobre resultados, historia, jugadores, etc.",
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Responde al comando /help."""
        user = update.effective_user
        logger.info(f"Comando /help recibido de {user.first_name} (ID: {user.id})")
        await update.message.reply_text(
            "🤖 ¿Cómo puedo ayudarte?\n"
            "Simplemente escríbeme tu pregunta sobre fútbol.\n\n"
            "Ejemplos:\n"
            "  - ¿Quién ganó la última Champions?\n"
            "  - Háblame de la historia de la Premier League\n"
            "  - ¿Cuándo juega la selección Colombia?\n\n"
            "Comandos:\n"
            "/start - Reiniciar conversación\n"
            "/help - Mostrar esta ayuda"
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Procesa los mensajes de texto recibidos."""
        message = update.message
        user = update.effective_user
        chat_id = update.effective_chat.id
        message_text = message.text

        if not message_text:
             logger.debug(f"Mensaje sin texto recibido de {user.first_name} (ID: {user.id})")
             return

        logger.info(f"Mensaje de texto recibido de {user.first_name} (ID: {user.id}): '{message_text}'")
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')

        response_text = "Lo siento, algo salió mal al procesar tu mensaje. 🤔 Intenta de nuevo."
        try:
            logger.debug(f"Llamando a predict_class para: '{message_text}'")
            intents_list = predict_class(message_text)
            logger.debug(f"Intenciones predichas: {intents_list}")

            logger.debug(f"Llamando a get_response para intenciones: {intents_list}")
            response_text = get_response(intents_list)
            logger.debug(f"Respuesta generada: '{response_text}'")

            if not response_text:
                response_text = "Hmm, no estoy seguro de cómo responder a eso. ¿Puedes reformular tu pregunta?"
                logger.warning(f"No se generó respuesta válida para: '{message_text}' (Intenciones: {intents_list})")

        except Exception as e:
            logger.error(f"Error en handle_message al procesar '{message_text}' de {user.id}: {e}", exc_info=True)
            # Mantenemos el mensaje de error genérico para el usuario

        await message.reply_text(response_text)
        logger.info(f"Respuesta enviada a {user.first_name} ({user.id}): '{response_text}'")

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Loggea errores no capturados por otros manejadores."""
        logger.error("Excepción no controlada al manejar un update:", exc_info=context.error)

    def _register_handlers(self):
        """Registra todos los manejadores en la aplicación."""
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        self.application.add_handler(MessageHandler(filters.COMMAND, self.unknown_command)) # Manejar comandos desconocidos
        self.application.add_error_handler(self.error_handler)
        logger.info("Manejadores de comandos, mensajes y errores registrados.")

    async def unknown_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Responde a comandos no reconocidos."""
        logger.warning(f"Comando desconocido recibido: {update.message.text}")
        await update.message.reply_text("Lo siento, no reconozco ese comando. Usa /help para ver los comandos disponibles.")


    def run(self):
        """Inicia el bot y comienza a escuchar actualizaciones de Telegram."""
        logger.info("Iniciando el bot (polling)... El bot está ahora en línea.")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)
        logger.info("El bot ha sido detenido.")

if __name__ == "__main__":
    logger.info("=============================================")
    logger.info("=== INICIANDO SCRIPT DEL BOT DE TELEGRAM ===")
    logger.info("=============================================")

    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_TOKEN:
        logger.critical("¡ERROR FATAL! La variable de entorno 'TELEGRAM_BOT_TOKEN' no está definida.")
        logger.critical("Verifica que exista, que el nombre sea correcto y reinicia la terminal/IDE.")
        sys.exit("Variable de entorno TELEGRAM_BOT_TOKEN no encontrada. El bot no puede iniciar.")
    else:
        logger.info(f"Token de Telegram cargado OK desde variable de entorno (inicia con: {TELEGRAM_TOKEN[:6]}...).")

    try:
        logger.info("Inicializando la instancia de TelegramBot...")
        telegram_bot = TelegramBot(telegram_token=TELEGRAM_TOKEN)
        logger.info("Llamando a telegram_bot.run()... (Presiona Ctrl+C para detener)")
        telegram_bot.run()

    except ValueError as ve:
        logger.critical(f"Error de configuración al crear TelegramBot: {ve}")
    except Exception as e:
        logger.critical(f"Error fatal inesperado durante la ejecución del bot: {e}")
        logger.critical(traceback.format_exc())
        sys.exit("Error fatal durante la ejecución del bot. Revisa los logs.")

    logger.info("============================================")
    logger.info("=== SCRIPT DEL BOT DE TELEGRAM FINALIZADO ===")
    logger.info("============================================")