from huggingface_hub import hf_hub_download
import torch
import os

import gradio as gr
from audioldm2 import text_to_audio, build_model
from share_btn import community_icon_html, loading_icon_html, share_js

os.environ["TOKENIZERS_PARALLELISM"] = "true"

default_checkpoint="audioldm2-full"
audioldm = None
current_model_name = None

def text2audio(
    text,
    guidance_scale,
    random_seed,
    n_candidates,
    model_name=default_checkpoint,
):
    global audioldm, current_model_name
    torch.set_float32_matmul_precision("high")

    if audioldm is None or model_name != current_model_name:
        audioldm = build_model(model_name=model_name)
        current_model_name = model_name
        audioldm = torch.compile(audioldm)

    # print(text, length, guidance_scale)
    waveform = text_to_audio(
        latent_diffusion=audioldm,
        text=text,
        seed=random_seed,
        duration=10,
        guidance_scale=guidance_scale,
        n_candidate_gen_per_text=int(n_candidates),
    )  # [bs, 1, samples]
    waveform = [
        gr.make_waveform((16000, wave[0]), bg_image="bg.png") for wave in waveform
    ]
    # waveform = [(16000, np.random.randn(16000)), (16000, np.random.randn(16000))]
    if len(waveform) == 1:
        waveform = waveform[0]
    return waveform

css = """
        a {
            color: inherit;
            text-decoration: underline;
        }
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: #000000;
            background: #000000;
        }
        input[type='range'] {
            accent-color: #000000;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        #advanced-btn {
            font-size: .7rem !important;
            line-height: 19px;
            margin-top: 12px;
            margin-bottom: 12px;
            padding: 2px 8px;
            border-radius: 14px !important;
        }
        #advanced-options {
            margin-bottom: 20px;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        #container-advanced-btns{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        #share-btn-container {
            display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
            margin-top: 10px;
            margin-left: auto;
        }
        #share-btn {
            all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;right:0;
        }
        #share-btn * {
            all: unset;
        }
        #share-btn-container div:nth-child(-n+2){
            width: auto !important;
            min-height: 0px !important;
        }
        #share-btn-container .wrap {
            display: none !important;
        }
        .gr-form{
            flex: 1 1 50%; border-top-right-radius: 0; border-bottom-right-radius: 0;
        }
        #prompt-container{
            gap: 0;
        }
        #generated_id{
            min-height: 700px
        }
        #setting_id{
          margin-bottom: 12px;
          text-align: center;
          font-weight: 900;
        }
"""
iface = gr.Blocks(css=css)

with iface:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 700px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <h1 style="font-weight: 900; margin-bottom: 7px; line-height: normal;">
                  AudioLDM 2: A General Framework for Audio, Music, and Speech Generation
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
                <a href="https://arxiv.org/abs/2301.12503">[Paper]</a>  <a href="https://audioldm.github.io/audioldm2">[Project page]</a> <a href="https://discord.com/invite/b64SEmdf">[Join Discord]</a>
              </p>
            </div>
        """
    )
    gr.HTML(
        """
        <p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings.
        <br/>
        <a href="https://huggingface.co/spaces/haoheliu/audioldm2-text2audio-text2music?duplicate=true">
        <img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
        <p/>
    """
    )
    with gr.Group():
        with gr.Box():
            ############# Input
            textbox = gr.Textbox(
                value="A forest of wind chimes singing a soothing melody in the breeze.",
                max_lines=1,
                label="Input your text here. Your text is important for the audio quality. Please ensure it is descriptive by using more adjectives.",
                elem_id="prompt-in",
            )

            with gr.Accordion("Click to modify detailed configurations", open=False):
                seed = gr.Number(
                    value=45,
                    label="Change this value (any integer number) will lead to a different generation result.",
                )
                # duration = gr.Slider(
                #     10, 10, value=10, step=2.5, label="Duration (seconds)"
                # )
                guidance_scale = gr.Slider(
                    0,
                    6,
                    value=3.5,
                    step=0.5,
                    label="Guidance scale (Large => better quality and relavancy to text; Small => better diversity)",
                )
                n_candidates = gr.Slider(
                    1,
                    3,
                    value=3,
                    step=1,
                    label="Automatic quality control. This number control the number of candidates (e.g., generate three audios and choose the best to show you). A Larger value usually lead to better quality with heavier computation",
                )
                # model_name = gr.Dropdown(
                #       ["audioldm-m-text-ft", "audioldm-s-text-ft", "audioldm-m-full","audioldm-s-full-v2", "audioldm-s-full", "audioldm-l-full"], value="audioldm-m-full", label="Choose the model to use. audioldm-m-text-ft and audioldm-s-text-ft are recommanded. -s- means small, -m- means medium and -l- means large",
                #   )
            ############# Output
            # outputs=gr.Audio(label="Output", type="numpy")
            outputs = gr.Video(label="Output", elem_id="output-video")

            # with gr.Group(elem_id="container-advanced-btns"):
            #   # advanced_button = gr.Button("Advanced options", elem_id="advanced-btn")
            #   with gr.Group(elem_id="share-btn-container"):
            #     community_icon = gr.HTML(community_icon_html, visible=False)
            #     loading_icon = gr.HTML(loading_icon_html, visible=False)
            #     share_button = gr.Button("Share to community", elem_id="share-btn", visible=False)
            # outputs=[gr.Audio(label="Output", type="numpy"), gr.Audio(label="Output", type="numpy")]
            btn = gr.Button("Submit").style(full_width=True)

        with gr.Group(elem_id="share-btn-container", visible=False):
            community_icon = gr.HTML(community_icon_html)
            loading_icon = gr.HTML(loading_icon_html)
            share_button = gr.Button("Share to community", elem_id="share-btn")

        # btn.click(text2audio, inputs=[
        #           textbox, duration, guidance_scale, seed, n_candidates, model_name], outputs=[outputs])
        btn.click(
            text2audio,
            inputs=[textbox, guidance_scale, seed, n_candidates],
            outputs=[outputs],
            api_name="text2audio",
        )

        share_button.click(None, [], [], _js=share_js)
        gr.HTML(
            """
        <div class="footer" style="text-align: center; max-width: 700px; margin: 0 auto;">
                    <p>Follow the latest update of AudioLDM 2 on our<a href="https://github.com/haoheliu/AudioLDM2" style="text-decoration: underline;" target="_blank"> Github repo</a>
                    </p>
                    <br>
                    <p>Model by <a href="https://twitter.com/LiuHaohe" style="text-decoration: underline;" target="_blank">Haohe Liu</a></p>
                    <br>
        </div>
        """
        )
        gr.Examples(
            [
                [
                    "An excited crowd cheering at a sports game.",
                    3.5,
                    45,
                    3,
                    default_checkpoint,
                ],
                [
                    "A cat is meowing for attention.",
                    3.5,
                    45,
                    3,
                    default_checkpoint,
                ],
                [
                    "Birds singing sweetly in a blooming garden.",
                    3.5,
                    45,
                    3,
                    default_checkpoint,
                ],
                [
                    "A modern synthesizer creating futuristic soundscapes.",
                    3.5,
                    45,
                    3,
                    default_checkpoint,
                ],
                [
                    "The vibrant beat of Brazilian samba drums.",
                    3.5,
                    45,
                    3,
                    default_checkpoint,
                ],
            ],
            fn=text2audio,
            # inputs=[textbox, duration, guidance_scale, seed, n_candidates, model_name],
            inputs=[textbox, guidance_scale, seed, n_candidates],
            outputs=[outputs],
            cache_examples=True,
        )
        gr.HTML(
            """
                <div class="acknowledgements">
                <p>Essential Tricks for Enhancing the Quality of Your Generated Audio</p>
                <p>1. Try to use more adjectives to describe your sound. For example: "A man is speaking clearly and slowly in a large room" is better than "A man is speaking". This can make sure AudioLDM 2 understands what you want.</p>
                <p>2. Try to use different random seeds, which can affect the generation quality significantly sometimes.</p>
                <p>3. It's better to use general terms like 'man' or 'woman' instead of specific names for individuals or abstract objects that humans may not be familiar with, such as 'mummy'.</p>
                </div>
                """
        )

        with gr.Accordion("Additional information", open=False):
            gr.HTML(
                """
                <div class="acknowledgments">
                    <p> We build the model with data from <a href="http://research.google.com/audioset/">AudioSet</a>, <a href="https://freesound.org/">Freesound</a> and <a href="https://sound-effects.bbcrewind.co.uk/">BBC Sound Effect library</a>. We share this demo based on the <a href="https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/375954/Research.pdf">UK copyright exception</a> of data for academic research. </p>
                            </div>
                        """
            )
# <p>This demo is strictly for research demo purpose only. For commercial use please <a href="haoheliu@gmail.com">contact us</a>.</p>

iface.queue(concurrency_count=3)
iface.launch(debug=True)
# iface.launch(debug=True, share=True)
