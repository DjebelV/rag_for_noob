import gradio as gr

def respond(message, history):
    
    print("\n===== HISTORY =====")
    print(history)
    print("===================\n")

    return f"Tu as demandé : {message}"

chatbot = gr.ChatInterface(respond)

if __name__ == "__main__":
    chatbot.launch()
