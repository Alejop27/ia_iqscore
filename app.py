import streamlit as st
# Ensure you have these functions defined in chatbot.py
from chatbot import predict_class, get_response
from datetime import datetime

# --- Configuración de la página ---
st.set_page_config(
    page_title="IQScore - Asistente Fútbol",
    page_icon="⚽",
    layout="wide"
)

# --- Paletas de colores mejoradas (inspiración moderna) ---
FOOTBALL_THEMES = {
    "Light": {
        "primary": "#1D4ED8",      # Azul más vibrante (Tailwind Blue 700)
        "secondary": "#60A5FA",    # Azul claro para gradiente (Tailwind Blue 400)
        "background": "#F3F4F6",  # Gris muy claro (Tailwind Gray 100)
        "content_bg": "#FFFFFF",  # Blanco
        "text": "#1F2937",      # Gris oscuro (Tailwind Gray 800)
        "accent": "#6B7280",      # Gris medio (Tailwind Gray 500)
        "user_bubble": "#3B82F6",  # Azul principal para usuario (Tailwind Blue 500)
        "user_text": "#FFFFFF",   # Texto blanco para burbuja usuario
        "bot_bubble": "#E5E7EB",   # Gris claro para bot (Tailwind Gray 200)
        "bot_text": "#1F2937",      # Texto oscuro para burbuja bot
        "header_text": "#FFFFFF",   # Texto blanco header
        "border": "#D1D5DB",      # Borde sutil (Tailwind Gray 300)
    },
    "Dark": {
        "primary": "#3B82F6",      # Azul brillante (Tailwind Blue 500)
        "secondary": "#1E40AF",    # Azul oscuro para gradiente (Tailwind Blue 800)
        "background": "#0f172a", # Azul noche muy oscuro (Tailwind Slate 900)
        "content_bg": "#1E293B",  # Azul grisáceo oscuro (Tailwind Slate 800)
        "text": "#E5E7EB",      # Gris claro (Tailwind Gray 200)
        "accent": "#64748B",      # Gris azulado (Tailwind Slate 500)
        "user_bubble": "#2563EB",  # Azul más intenso usuario (Tailwind Blue 600)
        "user_text": "#FFFFFF",   # Texto blanco
        "bot_bubble": "#334155",  # Azul grisáceo bot (Tailwind Slate 700)
        "bot_text": "#E5E7EB",      # Texto claro
        "header_text": "#FFFFFF",   # Texto blanco header
        "border": "#334155",      # Borde oscuro sutil (Tailwind Slate 700)
    }
}

# --- Estado de Sesión ---
if "theme" not in st.session_state:
    st.session_state.theme = "Dark" # Default to Dark

if "messages" not in st.session_state:
    st.session_state.messages = []
    # Initial bot message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "¡Hola! Soy IQScore, tu asistente de fútbol. ⚽ ¿En qué puedo ayudarte hoy? (Resultados, noticias, tácticas...)",
        "time": datetime.now().strftime("%H:%M")
    })

# --- Sidebar ---
with st.sidebar:
    # Simple logo or image
    st.image("https://cdn-icons-png.flaticon.com/512/53/53283.png", width=70) # Adjusted size

    # Use theme color for the title
    st.markdown(
        f"""
        <h2 style="color: {FOOTBALL_THEMES[st.session_state.theme]['primary']}; font-weight: 600;">
           IQScore Chatbot ⚽
        </h2>
        """, unsafe_allow_html=True
    )
    st.markdown("---")

    # Theme Selector
    selected_theme = st.radio(
        "🎨 Selecciona un Tema:",
        options=["Light", "Dark"],
        index=1 if st.session_state.theme == "Dark" else 0,
        key="theme_selector",
        horizontal=True,
    )
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()

    st.markdown("---")

    # Sidebar description with icons
    st.markdown(
        f"""
        <h3 style="color: {FOOTBALL_THEMES[st.session_state.theme]['primary']};">¿Qué puedo hacer?</h3>

        Soy tu asistente virtual de fútbol. Pregúntame sobre:
        <ul>
            <li>📊 Resultados y estadísticas</li>
            <li>📰 Noticias y fichajes</li>
            <li>♟️ Análisis tácticos</li>
            <li>🏆 Historia y récords</li>
            <li>⚽ ¡Y mucho más!</li>
        </ul>
        """, unsafe_allow_html=True
    )
    st.markdown("---")
    st.info("Escribe tu pregunta abajo y pulsa Enter.")
    st.caption(f"© {datetime.now().year} IQScore")

# --- Aplicar Tema Seleccionado ---
colors = FOOTBALL_THEMES[st.session_state.theme]
fixed_bubble_radius = 18 # Slightly larger radius

# --- CSS Personalizado Dinámico ---
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600&display=swap'); /* Changed to Inter for body text */

        /* Define CSS variables from theme */
        :root {{
            --primary: {colors['primary']};
            --secondary: {colors['secondary']};
            --background: {colors['background']};
            --content-bg: {colors['content_bg']};
            --text: {colors['text']};
            --accent: {colors['accent']};
            --user-bubble: {colors['user_bubble']};
            --user-text: {colors['user_text']};
            --bot-bubble: {colors['bot_bubble']};
            --bot-text: {colors['bot_text']};
            --header-text: {colors['header_text']};
            --border: {colors['border']};
            --bubble-radius: {fixed_bubble_radius}px;
        }}

        /* Global Styles */
        * {{ /* Apply box-sizing border-box universally */
             box-sizing: border-box;
             margin: 0;
             padding: 0;
        }}

        body {{
            font-family: 'Inter', sans-serif; /* Using Inter for better readability */
            background-color: var(--background);
            color: var(--text);
            line-height: 1.6;
        }}

        .stApp {{
            background-color: var(--background);
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] > div:first-child {{
            background-color: var(--content-bg);
            border-right: 1px solid var(--border); /* Use theme border color */
            padding-top: 1.5rem; /* Add some top padding */
        }}

        section[data-testid="stSidebar"] * {{
            color: var(--text); /* Ensure all sidebar text uses theme color */
        }}

        section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
            color: var(--primary); /* Titles in primary color */
        }}
        section[data-testid="stSidebar"] ul {{
            list-style-position: inside;
            padding-left: 0; /* Remove default padding */
            margin-top: 0.5rem;
        }}
        section[data-testid="stSidebar"] li {{
            margin-bottom: 0.3rem;
            font-size: 0.95rem;
        }}


        /* Header Styling */
        .gradient-header {{
            background: linear-gradient(120deg, var(--secondary) 0%, var(--primary) 100%); /* Smooth gradient */
            padding: 1.2rem 1.5rem; /* Adjusted padding */
            border-radius: 12px; /* Consistent radius */
            margin-bottom: 1.5rem; /* Space below header */
            color: var(--header-text);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15); /* Softer shadow */
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap; /* Allow wrapping on small screens */
        }}

        .header-title-group {{
             display: flex;
             align-items: center;
             gap: 15px; /* Space between logo and title */
        }}

         .header-icons-group {{
             display: flex;
             align-items: center;
             gap: 15px; /* Space between icons */
             margin-left: auto; /* Push icons to the right if space allows */
         }}

        .header-logo {{
            width: 45px; /* Slightly smaller logo */
            height: auto;
            filter: brightness(0) invert(1); /* Keep white */
        }}

        .title-font {{
            font-family: 'Bebas Neue', cursive;
            letter-spacing: 1.5px;
            font-size: 2.2rem; /* Adjusted size */
            color: var(--header-text);
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
            margin: 0; /* Remove default margin */
        }}

        .header-icon {{
             width: 35px; /* Slightly smaller icons */
             height: auto;
             filter: brightness(0) invert(1);
             opacity: 0.8; /* Slightly transparent for subtlety */
             transition: opacity 0.3s ease, transform 0.3s ease;
        }}
        .header-icon:hover {{
             opacity: 1;
             transform: scale(1.1); /* Slight grow on hover */
        }}


        /* Chat Container */
        .chat-container {{
            max-height: calc(100vh - 250px); /* Adjust height based on viewport, leave space for header/input */
            min-height: 400px; /* Ensure minimum height */
            overflow-y: auto;
            padding: 1rem 1.5rem; /* More padding */
            border-radius: 12px; /* Consistent radius */
            background-color: var(--content-bg);
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08); /* Softer shadow */
            border: 1px solid var(--border); /* Subtle border */
            scroll-behavior: smooth; /* Smooth scrolling */
        }}

        /* Chat Messages Base */
        .message-bubble {{
            padding: 0.8rem 1.2rem; /* Comfortable padding */
            border-radius: var(--bubble-radius);
            margin-bottom: 1rem; /* Space between bubbles */
            max-width: 85%; /* Slightly wider max-width */
            position: relative;
            word-wrap: break-word; /* Ensure long words break */
            overflow-wrap: break-word;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease; /* Smooth transition for hover */
        }}
        .message-bubble:hover {{
            transform: translateY(-3px) scale(1.01); /* Lift and slightly grow effect */
            box-shadow: 0 5px 12px rgba(0, 0, 0, 0.15); /* Enhanced shadow on hover */
        }}


        /* User Message */
        .user-message {{
            background-color: var(--user-bubble);
            color: var(--user-text);
            border-radius: var(--bubble-radius) var(--bubble-radius) 0 var(--bubble-radius); /* Rounded corners */
            float: right;
            clear: both;
            margin-left: auto; /* Align to right */
            animation: slideInRight 0.4s ease-out; /* Keep animation */
        }}

        /* Bot Message */
        .bot-message {{
            background-color: var(--bot-bubble);
            color: var(--bot-text);
            border-radius: var(--bubble-radius) var(--bubble-radius) var(--bubble-radius) 0; /* Rounded corners */
            float: left;
            clear: both;
            margin-right: auto; /* Align to left */
            animation: slideInLeft 0.4s ease-out; /* Keep animation */
        }}
         .bot-message .message-content {{ /* Style bot text */
            display: flex;
            align-items: flex-start; /* Align icon and text */
            gap: 8px; /* Space between icon and text */
         }}
         .bot-message .message-content .bot-icon {{ /* Style for bot icon */
            font-size: 1.1rem;
            line-height: 1.5; /* Match text line-height */
         }}


        /* Timestamp */
        .message-time {{
            font-size: 0.75rem; /* Slightly larger */
            color: var(--accent);
            margin-top: 0.4rem; /* More space above timestamp */
            display: block; /* Ensure it takes its own line */
            clear: both; /* Prevent overlapping floats */
            text-align: right; /* Default alignment */
        }}
        .bot-message .message-time {{
             text-align: left; /* Align bot timestamp left */
        }}
        .user-message .message-time {{
             text-align: right; /* Align user timestamp right */
             color: rgba(255, 255, 255, 0.7); /* Lighter timestamp for user bubble if needed */
             /* In dark mode, var(--accent) might be light enough */
             /* color: { 'rgba(0,0,0,0.5)' if st.session_state.theme == 'Light' else 'rgba(255,255,255,0.7)' }; */
        }}


        /* Chat Input Area */
        div[data-testid="stChatInput"] > div {{
            background-color: var(--content-bg); /* Match content background */
            border-top: 1px solid var(--border); /* Separator line */
            padding: 0.75rem 1rem; /* Adjust padding */
            box-shadow: 0 -4px 10px rgba(0, 0, 0, 0.05); /* Subtle top shadow */
        }}

        div[data-testid="stChatInput"] textarea {{
            border-radius: 25px !important; /* More rounded input */
            padding: 0.8rem 1.2rem !important; /* More padding */
            background-color: var(--background) !important; /* Slightly different bg for input */
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
            box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1); /* Inner shadow */
        }}
        div[data-testid="stChatInput"] textarea:focus {{
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px {colors['primary']}33 !important; /* Focus ring using primary color with transparency */
        }}

        /* Send Button */
        div[data-testid="stChatInput"] button {{
            border-radius: 50% !important;
            background-color: var(--primary) !important;
            color: var(--header-text) !important; /* White icon */
            width: 40px !important; /* Fixed size */
            height: 40px !important;
            padding: 0 !important; /* Remove default padding */
            margin-left: 0.5rem !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        div[data-testid="stChatInput"] button:hover {{
            background-color: {colors['secondary']} !important; /* Change color on hover */
            transform: scale(1.08); /* Slightly larger scale */
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }}
        div[data-testid="stChatInput"] button svg {{ /* Style the arrow icon */
             width: 20px;
             height: 20px;
        }}

        /* Custom Scrollbar */
        .chat-container::-webkit-scrollbar {{
            width: 8px;
            background-color: transparent;
        }}
        .chat-container::-webkit-scrollbar-thumb {{
            background-color: var(--primary);
            border-radius: 10px;
            border: 2px solid var(--content-bg); /* Creates padding around thumb */
        }}
        .chat-container::-webkit-scrollbar-track {{
            background-color: var(--background); /* Match app background */
            border-radius: 10px;
        }}

        /* Animations */
        @keyframes slideInRight {{
            from {{ transform: translateX(40px); opacity: 0; }}
            to {{ transform: translateX(0); opacity: 1; }}
        }}
        @keyframes slideInLeft {{
            from {{ transform: translateX(-40px); opacity: 0; }}
            to {{ transform: translateX(0); opacity: 1; }}
        }}

        /* Responsive Design */
        @media (max-width: 768px) {{
            .gradient-header {{
                padding: 1rem;
                flex-direction: column; /* Stack elements vertically */
                align-items: center; /* Center items */
                gap: 10px; /* Space when stacked */
            }}
            .header-title-group {{
                 justify-content: center; /* Center logo and title */
                 width: 100%;
            }}
             .header-icons-group {{
                 justify-content: center; /* Center icons */
                 width: 100%;
                 margin-left: 0; /* Remove auto margin */
             }}
            .title-font {{
                font-size: 1.8rem; /* Smaller title */
            }}
            .header-logo {{ width: 35px; }}
            .header-icon {{ width: 30px; }}

            .chat-container {{
                max-height: calc(100vh - 220px); /* Adjust for smaller screens */
                min-height: 300px;
                padding: 0.8rem;
            }}
            .message-bubble {{
                max-width: 92%; /* Allow bubbles to be wider */
                padding: 0.7rem 1rem;
            }}
            .message-time {{
                font-size: 0.7rem;
            }}

            div[data-testid="stChatInput"] > div {{
                padding: 0.5rem 0.75rem;
            }}
            div[data-testid="stChatInput"] textarea {{
                padding: 0.7rem 1rem !important;
            }}
             div[data-testid="stChatInput"] button {{
                 width: 36px !important;
                 height: 36px !important;
             }}
              div[data-testid="stChatInput"] button svg {{
                 width: 18px;
                 height: 18px;
             }}
        }}

         @media (max-width: 480px) {{
             .title-font {{
                font-size: 1.6rem; /* Even smaller */
             }}
             .message-bubble {{
                max-width: 95%;
            }}
         }}

    </style>
""", unsafe_allow_html=True)

# --- Header con temática de fútbol (mejorado) ---
st.markdown(
    f"""
    <div class="gradient-header">
        <div class="header-title-group">
            <img src="https://cdn-icons-png.flaticon.com/512/53/53283.png" class="header-logo">
            <h1 class="title-font">IQScore Football</h1>
        </div>
        <div class="header-icons-group">
            <img src="https://cdn-icons-png.flaticon.com/512/889/889648.png" class="header-icon" title="Live Scores">
            <img src="https://cdn-icons-png.flaticon.com/512/889/889649.png" class="header-icon" title="League Tables">
            <img src="https://cdn-icons-png.flaticon.com/512/889/889647.png" class="header-icon" title="Trophy Room">
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Mostrar el historial de mensajes (con contenedor y clases) ---
chat_display_container = st.container() # Use a container for better control if needed later

with chat_display_container:
    st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True) # Added ID for potential JS later

    for i, message in enumerate(st.session_state.messages):
        role_class = "user-message" if message["role"] == "user" else "bot-message"
        content = message["content"]
        time_str = message["time"]

        if message["role"] == "assistant":
             # Add a football icon consistently for bot messages using HTML structure
             st.markdown(
                f'''
                <div class="{role_class} message-bubble">
                    <div class="message-content">
                        <span class="bot-icon">⚽</span>
                        <span>{content}</span>
                    </div>
                    <div class="message-time">{time_str}</div>
                </div>
                ''',
                unsafe_allow_html=True
            )
        else:
             # User message structure
             st.markdown(
                f'''
                <div class="{role_class} message-bubble">
                    {content}
                    <div class="message-time">{time_str}</div>
                </div>
                ''',
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

# --- Input del usuario ---
if prompt := st.chat_input("Escribe tu consulta de fútbol..."):
    current_time = datetime.now().strftime("%H:%M")
    # Agregar mensaje del usuario al historial
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "time": current_time
    })

    # --- Simular respuesta del chatbot (reemplaza con tu lógica real) ---
    # Placeholder: Simulate prediction and response fetching
    try:
        intent_tag = predict_class(prompt)  # Predice la intención del usuario
        response = get_response(intent_tag) # Obtiene la respuesta correspondiente
    except Exception as e:
        st.error(f"Error processing chat: {e}") # Show error if chatbot logic fails
        response = "Lo siento, tuve un problema procesando tu solicitud. Por favor, intenta de nuevo."

    # Agregar respuesta del bot al historial
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "time": current_time
    })

    # Actualizar la interfaz para mostrar los nuevos mensajes
    st.rerun()