import gradio as gr

def greet(text, goal, part):
    return f"Text: {text}\nGoal: {goal}\nPart: {part}"

with gr.Blocks() as demo:
    gr.Markdown("# **📝AI-powered Academic Research Assistant📝**")
    gr.Markdown("### 🗒️ Description for AI-powered Academic Research Assistant")

    gr.Markdown('Write the text you what to expand or upload corresponding text file.')

    with gr.Tab('Write Text'):
        input_prompt = gr.Textbox(label='Initial Text',
                            placeholder='Write here your research text!',
                            lines=9,)
    with gr.Tab('Upload File'):
        txt_file = gr.File()
            
    gr.Markdown('✔️Fill parameters for your needs')
    with gr.Row(equal_height=True):
        request_goal = gr.Radio(label='Specify the purpose of your request.',
                                info="Pick one:",
                                choices=['Check Academic Style', 'Check Grammar', 'Write Text (Part)',],
                                value='Check Academic Style',)
        
        with gr.Accordion("In case you need to Write Text (Part) choose appropriate option!", open=False):
            part_to_write = gr.CheckboxGroup(label='What part for Assistant to write?', 
                                            info='Here you need to specify what part of your research you need to complete.\n You may chose as many as needed:',
                                            choices=['Full Text', 'Abstract', 'Introduction', 'Methodology', 'Discussion', 'Conclusion'],
                                            value='Full Text',)
    
    with gr.Row(equal_height=True):
        submit_btn = gr.Button('Confirm!')
        clear_btn = gr.Button('Clear')

    gr.Markdown('##### 📌Assistant Responce')
    responce = gr.Textbox(label="Generated Text", 
                          lines=9, 
                          placeholder='Here the generated text will appear!', 
                          show_copy_button=True)
    
    
    submit_btn.click(fn=greet, inputs=[input_prompt, request_goal, part_to_write], outputs=[responce])
    clear_btn.click(lambda: (None, None, None, None), None, 
                    outputs=[input_prompt, request_goal, part_to_write, responce])

demo.launch()
