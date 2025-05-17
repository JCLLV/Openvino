#-------- Mayo de 2025. JC Llanos V. -----
#-------- jcllanosv@hotmail.com -----
#-------- https://www.linkedin.com/in/jcllanosv/
import time
import os
import warnings
import traceback
import fitz  # PyMuPDF
import tkinter as tk
from tkinter import filedialog
from openvino.runtime import Core
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
from huggingface_hub import logging as hf_logging
from duckduckgo_search import DDGS

# --- Configuraciones Iniciales ---
warnings.filterwarnings("ignore", category=DeprecationWarning, module='openvino.runtime')
hf_logging.set_verbosity_warning() # Muestra advertencias y errores, pero no logs INFO de hf_hub

LOCAL_OPENVINO_MODELS_DIR = "mis_modelos_openvino"

# --- Modelos Disponibles ---
# Formato: (Nombre para mostrar, ID de Hugging Face, necesita_trust_remote_code)
# Priorizamos versiones cuantizadas (INT8, INT4) para mejor rendimiento en iGPU
AVAILABLE_MODELS = [
    # Modelos del namespace OpenVINO (ya optimizados)
    ("OpenVINO: TinyLlama 1.1B Chat (INT8)", "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int8-ov", False),
    ("OpenVINO: TinyLlama 1.1B Chat (INT4)", "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov", False),
    ("OpenVINO: TinyLlama 1.1B Chat (FP16)", "OpenVINO/TinyLlama-1.1B-Chat-v1.0-fp16-ov", False),

    ("OpenVINO: Phi-2 (INT8)", "OpenVINO/phi-2-int8-ov", True), # Phi a menudo necesita trust_remote_code
    ("OpenVINO: Phi-2 (INT4)", "OpenVINO/phi-2-int4-ov", True),
    ("OpenVINO: Phi-2 (FP16)", "OpenVINO/phi-2-fp16-ov", True),
    
    # --- Modelos Especializados en Código (INT4 es ideal para iGPU) ---
    ("OpenVINO: StarCoder2 3B (INT4, Código)", "OpenVINO/starcoder2-3b-int4-ov", False),
    ("OpenVINO: StarCoder2 3B (INT8, Código)", "OpenVINO/starcoder2-3b-int8-ov", False),
    # ("OpenVINO: StarCoder2 7B (INT4, Código) - Grande", "OpenVINO/starcoder2-7b-int4-ov", False), # 7B puede ser pesado

    ("OpenVINO: CodeGen2 3.7B P (INT4, Código)", "OpenVINO/codegen2-3_7B_P-int4-ov", False), # "P" a menudo indica Python, pero son multilingües
    ("OpenVINO: CodeGen2 3.7B P (INT8, Código)", "OpenVINO/codegen2-3_7B_P-int8-ov", False),
    
    # Nuevos modelos DeepSeek-R1-Distill-Qwen
    ("OpenVINO: DeepSeek-Qwen 1.5B (INT8)", "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int8-ov", False), # Qwen base puede necesitar True, pero el -ov podría no. Probar.
    ("OpenVINO: DeepSeek-Qwen 1.5B (INT4)", "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-ov", False),
    ("OpenVINO: DeepSeek-Qwen 1.5B (FP16)", "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-fp16-ov", False),

    ("OpenVINO: DeepSeek-Qwen 7B (INT8)", "OpenVINO/DeepSeek-R1-Distill-Qwen-7B-int8-ov", False), # Precaución con 7B
    ("OpenVINO: DeepSeek-Qwen 7B (INT4)", "OpenVINO/DeepSeek-R1-Distill-Qwen-7B-int4-ov", False), # Precaución con 7B
    # ("OpenVINO: DeepSeek-Qwen 7B (FP16)", "OpenVINO/DeepSeek-R1-Distill-Qwen-7B-fp16-ov", False), # 7B FP16 muy pesado

    # ("OpenVINO: DeepSeek-Qwen 14B (INT8) - ¡MUY GRANDE!", "OpenVINO/DeepSeek-R1-Distill-Qwen-14B-int8-ov", False), # Extremadamente grande
    # ("OpenVINO: DeepSeek-Qwen 14B (INT4) - ¡MUY GRANDE!", "OpenVINO/DeepSeek-R1-Distill-Qwen-14B-int4-ov", False), # Extremadamente grande


    ("OpenVINO: Mistral 7B Instruct v0.1 (INT8)", "OpenVINO/mistral-7b-instruct-v0.1-int8-ov", False),
    ("OpenVINO: Mistral 7B Instruct v0.1 (INT4)", "OpenVINO/mistral-7b-instruct-v0.1-int4-ov", False),
    # ("OpenVINO: Mistral 7B Instruct v0.1 (FP16)", "OpenVINO/mistral-7b-instruct-v0.1-fp16-ov", False), # 7B FP16 puede ser pesado

    ("OpenVINO: Zephyr 7B Beta (INT8)", "OpenVINO/zephyr-7b-beta-int8-ov", False),
    ("OpenVINO: Zephyr 7B Beta (INT4)", "OpenVINO/zephyr-7b-beta-int4-ov", False),

    ("OpenVINO: Dolly v2 3B (INT8)", "OpenVINO/dolly-v2-3b-int8-ov", False), # Dolly es más antiguo pero conocido
    ("OpenVINO: Dolly v2 3B (INT4)", "OpenVINO/dolly-v2-3b-int4-ov", False),

    ("OpenVINO: RedPajama INCITE Chat 3B (INT8)", "OpenVINO/RedPajama-INCITE-Chat-3B-v1-int8-ov", False),
    ("OpenVINO: RedPajama INCITE Chat 3B (INT4)", "OpenVINO/RedPajama-INCITE-Chat-3B-v1-int4-ov", False),

    ("OpenVINO: Gemma 2B IT (INT8, Google)", "OpenVINO/gemma-2b-it-int8-ov", False), # "it" = Instruction Tuned
    ("OpenVINO: Gemma 2B IT (INT4, Google)", "OpenVINO/gemma-2b-it-int4-ov", False),

    ("OpenVINO: Qwen2 0.5B Instruct (INT8, Alibaba)", "OpenVINO/Qwen2-0.5B-Instruct-int8-ov", True), # Qwen a veces necesita trust_remote_code
    ("OpenVINO: Qwen2 0.5B Instruct (INT4, Alibaba)", "OpenVINO/Qwen2-0.5B-Instruct-int4-ov", True),
    ("OpenVINO: Qwen2 1.5B Instruct (INT8, Alibaba)", "OpenVINO/Qwen2-1.5B-Instruct-int8-ov", True),
    ("OpenVINO: Qwen2 1.5B Instruct (INT4, Alibaba)", "OpenVINO/Qwen2-1.5B-Instruct-int4-ov", True),

    # Modelos originales de Hugging Face (requerirán conversión por Optimum la primera vez)
    ("HuggingFace: DistilGPT-2 (muy pequeño)", "distilgpt2", False),
    ("HuggingFace: TinyLlama 1.1B Chat (original)", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", False),
    ("HuggingFace: Microsoft Phi-1.5 (1.5B)", "microsoft/phi-1_5", True),

    # Modelos MUY GRANDES (pueden fallar o ser extremadamente lentos en iGPU)
    # ("OpenVINO: Mixtral-8x7B Instruct (INT4) - ¡MUY GRANDE!", "OpenVINO/mixtral-8x7b-instruct-v0.1-int4-ov", False),
    # ("OpenVINO: GPT-J 6B (INT4) - GRANDE", "OpenVINO/gpt-j-6b-int4-ov", False),
]

DEVICE = "GPU"
MAX_NEW_TOKENS = 500
DO_SAMPLE = True
TEMPERATURE = 0.1
TOP_K = 40
TOP_P = 0.9
ENABLE_INTERNET_SEARCH = True
MAX_SEARCH_RESULTS = 2
SEARCH_REGION = 'cl-es' # Búsqueda en contexto de Chile.
SEARCH_TIMELIMIT = 'y'
MAX_HISTORIAL_TURNOS = 3
CURRENT_PDF_TEXT = None
CURRENT_PDF_NAME = None
MAX_PDF_CHARS_FOR_PROMPT = 20000

# --- Funciones Auxiliares ---
def verificar_dispositivos_openvino():
    try:
        core = Core()
        dispositivos_disponibles = core.available_devices
        print(f"OpenVINO - Dispositivos disponibles: {dispositivos_disponibles}")
        if DEVICE not in dispositivos_disponibles and DEVICE.split('.')[0] not in dispositivos_disponibles:
            print(f"Advertencia: Dispositivo '{DEVICE}' no encontrado explícitamente.")
        else:
            print(f"OpenVINO - Se intentará usar: {DEVICE}")
        del core
    except Exception as e:
        print(f"Error al verificar dispositivos OpenVINO: {e}")

def seleccionar_modelo():
    print("-" * 70)
    print("Selecciona un Modelo LLM para OpenVINO:")
    print("-" * 70)
    for i, (nombre_display, model_id_hf, _) in enumerate(AVAILABLE_MODELS):
        display_model_id = model_id_hf[:45] + '...' if len(model_id_hf) > 45 else model_id_hf
        print(f"{i+1:2d}. {nombre_display:<50} (ID: {display_model_id})")
    print("-" * 70)
    while True:
        try:
            choice_str = input(f"Elige un modelo (1-{len(AVAILABLE_MODELS)}) o escribe 'salir': ").strip()
            if choice_str.lower() in ["salir", "exit", "quit"]: return None, None, None
            choice = int(choice_str) - 1
            if 0 <= choice < len(AVAILABLE_MODELS):
                nombre_display, model_id_hf, necesita_trust = AVAILABLE_MODELS[choice]
                print(f"\nHas seleccionado: {nombre_display}")
                return model_id_hf, necesita_trust, nombre_display
            else: print("Opción no válida.")
        except ValueError: print("Entrada no válida. Ingresa un número.")
        except Exception as e: print(f"Error en selección: {e}"); return None, None, None

def buscar_en_internet(query: str):
    if not ENABLE_INTERNET_SEARCH: return None
    print(f"Buscando en internet sobre: \"{query}\"...")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query, region=SEARCH_REGION, safesearch='moderate',
                timelimit=SEARCH_TIMELIMIT, max_results=MAX_SEARCH_RESULTS
            ))
        if results:
            contexto = "\n### Contexto de Internet Relevante:\n"
            for i, res in enumerate(results):
                contexto += f"{i+1}. Título: {res.get('title', 'N/A')}\n   Fragmento: {res.get('body', 'N/A')}\n\n"
            return contexto.strip()
        else: print("No se encontraron resultados relevantes en internet."); return None
    except Exception as e: print(f"Error durante la búsqueda en internet: {e}"); return None

def seleccionar_archivo_pdf_con_dialogo():
    root = tk.Tk()
    root.withdraw()
    filetypes = (('Archivos PDF', '*.pdf'), ('Todos los archivos', '*.*'))
    pdf_path = filedialog.askopenfilename(
        title='Selecciona un archivo PDF',
        initialdir=os.getcwd(),
        filetypes=filetypes
    )
    root.destroy()
    return pdf_path if pdf_path else None

def cargar_y_extraer_texto_pdf(pdf_path):
    global CURRENT_PDF_TEXT, CURRENT_PDF_NAME
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text("text") for page in doc)
        doc.close()
        if not text.strip():
            print(f"Advertencia: PDF '{os.path.basename(pdf_path)}' sin texto extraíble o vacío.")
            CURRENT_PDF_TEXT = None; CURRENT_PDF_NAME = None; return False
        CURRENT_PDF_TEXT = text
        CURRENT_PDF_NAME = os.path.basename(pdf_path)
        char_count = len(CURRENT_PDF_TEXT); word_count = len(CURRENT_PDF_TEXT.split())
        print(f"PDF '{CURRENT_PDF_NAME}' cargado. Caracteres: {char_count}, Palabras aprox.: {word_count}")
        if char_count > MAX_PDF_CHARS_FOR_PROMPT:
            print(f"INFO: PDF largo. Se usarán los primeros ~{MAX_PDF_CHARS_FOR_PROMPT} caracteres para el contexto.")
        return True
    except Exception as e:
        print(f"Error al cargar/leer PDF '{pdf_path}': {e}")
        CURRENT_PDF_TEXT = None; CURRENT_PDF_NAME = None; return False

# --- Función Principal del Chatbot ---
def iniciar_chat(model_id_to_load, trust_remote_code_flag, model_display_name):
    global CURRENT_PDF_TEXT, CURRENT_PDF_NAME

    if not model_id_to_load:
        print("ID de modelo no proporcionado."); return

    verificar_dispositivos_openvino()
    os.makedirs(LOCAL_OPENVINO_MODELS_DIR, exist_ok=True)
    safe_model_name_for_path = model_id_to_load.replace("/", "__")
    local_model_dir = os.path.join(LOCAL_OPENVINO_MODELS_DIR, safe_model_name_for_path)

    ov_model = None
    tokenizer = None

    expected_ov_model_xml = os.path.join(local_model_dir, "openvino_model.xml")
    expected_tokenizer_config = os.path.join(local_model_dir, "tokenizer_config.json")

    if os.path.exists(local_model_dir) and \
       os.path.exists(expected_ov_model_xml) and \
       os.path.exists(expected_tokenizer_config):
        print(f"\nArchivos de modelo y tokenizador encontrados en: {local_model_dir}")
        print("Intentando cargar desde directorio local...")
        try:
            ov_model = OVModelForCausalLM.from_pretrained(
                local_model_dir, device=DEVICE, trust_remote_code=trust_remote_code_flag
            )
            if hasattr(ov_model, 'tokenizer') and ov_model.tokenizer is not None:
                tokenizer = ov_model.tokenizer
                print(f"¡Modelo y tokenizador '{model_display_name}' cargados desde local!")
            else: # Fallback si Optimum no adjuntó el tokenizador
                print(f"Modelo '{model_display_name}' cargado localmente, tokenizador no adjunto. Cargando tokenizador por separado...")
                tokenizer = AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=trust_remote_code_flag)
                if hasattr(ov_model, 'tokenizer'): ov_model.tokenizer = tokenizer
                print(f"Tokenizador para '{model_display_name}' cargado desde local.")
        except Exception as e:
            print(f"Error al cargar desde local '{local_model_dir}': {e}")
            # print(f"Detalles: {traceback.format_exc()}")
            print("Se intentará desde Hugging Face Hub.")
            ov_model = None; tokenizer = None
    
    if ov_model is None: # Si no se cargó desde local o falló
        print(f"\nIniciando carga desde Hugging Face Hub para '{model_display_name}' ({model_id_to_load})")
        if tokenizer is None: # Cargar tokenizador si no se pudo antes
            print("Cargando tokenizador desde Hugging Face Hub...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id_to_load, trust_remote_code=trust_remote_code_flag)
                print("Tokenizador cargado desde Hugging Face Hub.")
            except Exception as e: print(f"Error fatal al cargar tokenizador: {e}"); return

        print(f"Cargando modelo '{model_display_name}' desde Hugging Face Hub...")
        export_needed = not model_id_to_load.startswith("OpenVINO/")
        load_message = "Convirtiendo a OpenVINO IR..." if export_needed else "Descargando (si es necesario)..."
        print(load_message)
        try:
            ov_model = OVModelForCausalLM.from_pretrained(
                model_id_to_load, device=DEVICE, export=export_needed, 
                trust_remote_code=trust_remote_code_flag, tokenizer=tokenizer
            )
            if hasattr(ov_model, 'tokenizer') and ov_model.tokenizer: tokenizer = ov_model.tokenizer
            elif tokenizer: ov_model.tokenizer = tokenizer
            else: print("ERROR CRÍTICO: No se pudo establecer tokenizador."); return
            print(f"¡Modelo '{model_display_name}' cargado/convertido desde Hugging Face!")
            os.makedirs(local_model_dir, exist_ok=True)
            print(f"Guardando en: {local_model_dir}...")
            try:
                ov_model.save_pretrained(local_model_dir)
                print(f"Modelo y tokenizador guardados en '{local_model_dir}'.")
                # print(f"Contenido: {os.listdir(local_model_dir)}")
            except Exception as e_save: print(f"Error al guardar: {e_save}") # print(f"Detalles: {traceback.format_exc()}")
        except Exception as e_hub: print(f"Error al cargar/convertir desde Hub: {e_hub}"); print(f"Detalles: {traceback.format_exc()}"); return

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"INFO: tokenizer.pad_token_id no definido, usando eos_token_id: {tokenizer.eos_token_id}")
        else: print("ADVERTENCIA CRÍTICA: Falta pad_token_id y eos_token_id.")

    CURRENT_PDF_TEXT = None; CURRENT_PDF_NAME = None # Reiniciar PDF para la sesión
    historial_conversacion = []
    print(f"\nListo para chatear con {model_display_name}. Escribe 'salir' para terminar.")
    print("Comandos PDF: !cargar_pdf (abrirá diálogo), !resumir_pdf, !comentar_pdf, !olvidar_pdf")

    while True:
        user_prompt_original = input("\nTú: ").strip()
        if user_prompt_original.lower() in ["salir", "exit", "quit"]: print("Saliendo..."); break
        if not user_prompt_original: continue

        if user_prompt_original.lower() == "!cargar_pdf":
            print("Abriendo diálogo para seleccionar PDF...")
            pdf_path_input = seleccionar_archivo_pdf_con_dialogo()
            if pdf_path_input:
                if os.path.exists(pdf_path_input) and pdf_path_input.lower().endswith(".pdf"):
                    print(f"Intentando cargar PDF: {pdf_path_input}")
                    cargar_y_extraer_texto_pdf(pdf_path_input)
                else: print(f"Ruta de PDF no válida: '{pdf_path_input}'")
            else: print("No se seleccionó PDF.")
            continue
        elif user_prompt_original.lower() == "!olvidar_pdf":
            if CURRENT_PDF_NAME: print(f"PDF '{CURRENT_PDF_NAME}' olvidado."); CURRENT_PDF_TEXT = None; CURRENT_PDF_NAME = None
            else: print("No hay PDF cargado para olvidar.")
            continue
        
        prompt_para_historial = "".join([f"Humano: {u}\nAsistente: {m}\n" for u, m in historial_conversacion[-MAX_HISTORIAL_TURNOS:]])
        final_prompt_for_llm = prompt_para_historial
        generated_user_query_for_llm = user_prompt_original

        contexto_principal_usado = "" # Para saber qué tipo de contexto se usó

        if CURRENT_PDF_TEXT and \
           (any(cmd in user_prompt_original.lower() for cmd in ["!resumir_pdf", "!comentar_pdf"]) or \
            any(term in user_prompt_original.lower() for term in ["pdf", "documento", "texto cargado"])):
            
            pdf_extract_to_use = CURRENT_PDF_TEXT[:MAX_PDF_CHARS_FOR_PROMPT]
            context_source_info = f"\n### Contexto del PDF '{CURRENT_PDF_NAME}' (primeras ~{len(pdf_extract_to_use)} caracteres):\n"
            if len(CURRENT_PDF_TEXT) > MAX_PDF_CHARS_FOR_PROMPT :
                 print(f"INFO: Usando las primeras ~{MAX_PDF_CHARS_FOR_PROMPT} caracteres del PDF '{CURRENT_PDF_NAME}'.")
            
            final_prompt_for_llm += context_source_info + pdf_extract_to_use + "\n\n"
            contexto_principal_usado = "PDF"
            
            if user_prompt_original.lower() == "!resumir_pdf":
                generated_user_query_for_llm = "Por favor, proporciona un resumen conciso del texto del PDF previamente proporcionado."
            elif user_prompt_original.lower() == "!comentar_pdf":
                generated_user_query_for_llm = "Por favor, analiza y proporciona comentarios sobre los puntos clave del texto del PDF previamente proporcionado."
        
        elif ENABLE_INTERNET_SEARCH:
            internet_context_str = buscar_en_internet(user_prompt_original)
            if internet_context_str:
                final_prompt_for_llm += f"{internet_context_str}\n\n"
                contexto_principal_usado = "Internet"
        
        if contexto_principal_usado:
             print(f"--- (Usando contexto de {contexto_principal_usado}) ---")
        else:
             print("--- (Sin contexto adicional externo) ---")

        final_prompt_for_llm += f"Humano: {generated_user_query_for_llm}\nAsistente:"
        
        tokenizer_defined_max_len = getattr(tokenizer, 'model_max_length', None)
        practical_max_len_cap = 4096 
        if tokenizer_defined_max_len is None or \
           not isinstance(tokenizer_defined_max_len, int) or \
           tokenizer_defined_max_len > practical_max_len_cap:
            model_max_input_length = practical_max_len_cap
        else:
            model_max_input_length = tokenizer_defined_max_len
        
        inputs = tokenizer(final_prompt_for_llm, return_tensors="pt", truncation=True, max_length=model_max_input_length)
        input_ids = inputs.input_ids; attention_mask = inputs.attention_mask
        
        print(f"{model_display_name} pensando...")
        start_time = time.time()
        try:
            gen_kwargs = {
                "attention_mask": attention_mask, "max_new_tokens": MAX_NEW_TOKENS, "do_sample": DO_SAMPLE,
            }
            if tokenizer.pad_token_id is not None: gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
            elif tokenizer.eos_token_id is not None: gen_kwargs["pad_token_id"] = tokenizer.eos_token_id
            if DO_SAMPLE:
                gen_kwargs["temperature"] = TEMPERATURE; gen_kwargs["top_k"] = TOP_K; gen_kwargs["top_p"] = TOP_P
            
            output_ids = ov_model.generate(input_ids, **gen_kwargs)
            end_time = time.time(); tiempo_inferencia = end_time - start_time
            input_token_length = input_ids.shape[1]
            generated_token_ids = output_ids[0][input_token_length:]
            response_only = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()
            
            print(f"\n{model_display_name} (en {tiempo_inferencia:.2f}s): {response_only}")
            historial_conversacion.append((generated_user_query_for_llm, response_only))
            if len(historial_conversacion) > MAX_HISTORIAL_TURNOS : 
                historial_conversacion = historial_conversacion[-MAX_HISTORIAL_TURNOS:]
        except Exception as e:
            print(f"Error durante la generación: {e}"); print(f"Detalles: {traceback.format_exc()}")

# --- Punto de Entrada del Script ---
if __name__ == "__main__":
    model_to_run_id, model_trust_remote, display_name = seleccionar_modelo()
    if model_to_run_id:
        iniciar_chat(model_to_run_id, model_trust_remote, display_name)
    else:
        print("No se seleccionó ningún modelo. Saliendo del programa.")
