import streamlit as st
import logging
logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

from transformers import AutoTokenizer, AutoModelForCausalLM
import tokenizers

from aitextgen import aitextgen

@st.cache(hash_funcs={tokenizers.Tokenizer: lambda _: None,
                      tokenizers.AddedToken: lambda _: None})
def load_generation_model():
  return aitextgen(model = "eastmountaincode/duneGenerationNoUser")

ai = load_generation_model()

from keras.models import load_model
import autokeras as ak

@st.cache(allow_output_mutation=True)
def load_rating_model():
  return load_model('rating/content/saved_model2',
                          custom_objects=ak.CUSTOM_OBJECTS)
model_rating = load_rating_model()

@st.cache(allow_output_mutation=True)
def load_helpful_model():
  return load_model('helpful/content/saved_model/my_model',
                           custom_objects=ak.CUSTOM_OBJECTS)
model_helpful = load_helpful_model()

st.image("duneGeneratorPic-01.png")

def generateReview():
  validText = False
  while(validText == False):
    print("Generating")
    text = ai.generate_one(
                prompt="BODY:",
                max_length=200,
                top_p=0.9,
                temperature = 1.0)
    textList = text.split("\n")
    print(textList)
    if len(textList) < 3:
      print("Don't have all three parts")
      continue
    output = ["", "", ""]
    output[0] = textList[0][6:]
    output[1] = textList[1][7:]
    validText = True
  textForClassification = "TITLE: " + output[1] + "\n" + "BODY: " + output[0]
  predicted_rating = int(round(model_rating.predict([textForClassification]).item(), 1))
  predicted_helpful = int(round(model_helpful.predict([textForClassification]).item(), 1))

  generatedData = {}
  generatedData["Title"] = output[1]
  generatedData["Review"] = output[0]
  generatedData["Rating"] = predicted_rating
  generatedData["HelpfulScore"] = predicted_helpful

  st.header(generatedData["Title"])
  st.caption("Rating: " + str(generatedData["Rating"]) + "/10")
  st.write(generatedData["Review"])
  st.caption(str(generatedData["HelpfulScore"]) + " percent of people found this review helpful")

    

def generateReview2():
  validText = False
  while(validText == False):
    print("Generating")
    text = ai.generate_one(
                prompt="BODY:",
                max_length=200,
                top_p=0.9,
                temperature = 1.0,
                num_beams = 2,
                repetition_penalty = 3.0)
    textList = text.split("\n")
    print(textList)
    if len(textList) < 3:
      print("Don't have all three parts")
      continue
    output = ["", "", ""]
    output[0] = textList[0][6:]
    output[1] = textList[1][7:]
    validText = True
    
  textForClassification = "TITLE: " + output[1] + "\n" + "BODY: " + output[0]
  predicted_rating = int(round(model_rating.predict([textForClassification]).item(), 1))
  predicted_helpful = int(round(model_helpful.predict([textForClassification]).item(), 1))

  generatedData = {}
  generatedData["Title"] = output[1]
  generatedData["Review"] = output[0]
  generatedData["Rating"] = predicted_rating
  generatedData["HelpfulScore"] = predicted_helpful

  st.header(generatedData["Title"])
  st.caption("Rating: " + str(generatedData["Rating"]) + "/10")
  st.write(generatedData["Review"])
  st.caption(str(generatedData["HelpfulScore"]) + " percent of people found this review helpful")


##generateButton = st.button("Generate review", on_click = generateReview)
generateButton2 = st.button("Generate review",
                            on_click = generateReview2)
  

