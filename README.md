<table>
  <tr>
    <td>
      <img src="src/images/persona_chatbot_logo.png" alt="Project Logo" width="200"/>
    </td>
    <td>
      <h1>PERSONA CHATBOT PROJECT</h1>
      <p>A Streamlit-based chatbot that mimics the conversational style of various public figures (Andy Warhol, Jean-Michel Basquiat, Kurt Cobain, Jim Morrison) using AI models trained on curated datasets.</p>
    </td>
  </tr>
</table>

## Disclaimer & Ethical Use

This project features chatbots designed to mimic the conversational style and persona of various public figures based on publicly available data and creative interpretation. These chatbots do **not** represent the real thoughts, beliefs, or opinions of the individuals depicted. All responses are generated by AI and are for educational, experimental, or entertainment purposes only.

**Please use responsibly.**  
Do not present these chatbots or their outputs as genuine statements from the actual persons.

If you have ethical concerns or believe any content is inappropriate, please open an issue or contact the repository maintainer.

## Datasets

During training the following datasets were used (not available on GitHub):
- `warhol_data.csv`
- `basquiat_data.csv`
- `cobain_data.csv`
- `morrison_data.csv`
- `all_persona_qa.jsonl`

Each CSV has the following columns:
- **Source**: Origin of the quote/text (e.g. Book, Interview)
- **Type**: Subtype (e.g. Diary Entry, Quote)
- **Content**: Main text (used for model training/chat)
- **Category**: Topic (e.g. Art, Business)
- **Conversational_Style**: Tone or style
- **Person**: Persona tag (e.g. AW, JMB, KC, JM)

## Training

Models (ChatGPT-NEO 125M) are trained on the `Content` field, optionally using other columns for style conditioning.

## Chatbot

To run the chatbot:

```sh
streamlit run app.py
```
Or see: 

https://apch-persona-chatbot.streamlit.app


## Model Files

**Note:** The model files are not included in this repository due to their size.
- Contact the repository owner for access to the model files.
