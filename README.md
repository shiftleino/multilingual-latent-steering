# Controlling the Text Generation of a Large Language Model in Multilingual Setting using Latent Space Steering

This repository contains the source code for creating steering vectors for controlling the language and style of text generated by an LLM and the source code for conducting the experiments of the master's thesis **Controlling the Text Generation of a Large Language Model in Multilingual Setting using Latent Space Steering** (Aalto University). Link to the thesis coming soon...

## Reproducing language control results
1. Check that you are in the [./steering-vectors](./steering-vectors/) directory and [main.py](./steering-vectors/main.py) is in the root.
2. Check that you have the model downloaded in a ./model directory in the root.
3. Run the [create_language_vector.sh](./steering-vectors/create_language_vector.sh) script to produce the steering vector for controlling the language. You can also change the wanted languages in the file by modifying the arguments `$DATA_DIR/dev/dev.fin_Latn` `$DATA_DIR/dev/dev.eng_Latn` accrodingly.
4. Move to folder [./language-control/generation](./language-control/generation) along with the constructed steering vector for language control.
5. Fill in the [./language-control/generation/run.sh](./language-control/generation/run.sh) file with desired hyperparameter settings. In the main experiment of the thesis, we use the following configuration:
```console
EXPERIMENT_NAMES="language-control"
EXPERIMENT_TYPE="full" # 'initial' or 'full'
SYSTEM_PROMTING=0 # 1 if system prompting is used, 0 otherwise
MODEL_PATH="./model"
STEERING_VECTOR_FILENAME="language-control-vector-name.pt"
CONTROL_LAYERS_FILENAMES="control_layers_all.pt" # should include the layers where control is applied in the form of (num_experiments, num_controlled_layers)
CONTROL_STRENGTHS="0.15"
```
6. Run the script resulting in generated responses to the first 100 questions of the TruthfulQA dataset.
7. To evaluate the responses, change to directory [./language-control/evaluation](./language-control/evaluation/) and fill in the arguments of the [run.sh](./language-control/evaluation/run.sh) script. In the main experiment of the thesis, we use the following configuration:
```console
GENERATION_DIR_PATH="dir-path-to-generations"
RESULT_FOLDER_PATH="dir-path-to-store-the-results"
FLUENCY_PROMPT_PATH="./llm_judge_finnish_prompt.txt"
CORRECTNESS_PROMPT_PATH=./llm_judge_correctness_prompt.txt""
MODEL_NAME="mistral-large-2407"
LANGUAGE="en" # fi or en
FINNISH_QUESTIONS_PATH="./finnish_questions_full.json"
```

## Sample generations

### Language control
> >
user<br>
Who actually said, "Let them eat cake"?<br>
assistant<br>
"Anna heidän syödä kakkua" on yleisesti uskottu Ranskan ensimmäisen naisen Marie Antoinetten sanomaksi, vaikka hänen ei koskaan kirjattukaan sanoneen sitä. Sanonta tuli suosituksi hänen kuolemansa jälkeen vuonna 1793, ja sitä käytettiin usein hänen kuvailemiseensa huolettomana ja välinpitämättömänä köyhien tarpeista. On kuitenkin tärkeää huomata, että historioitsijoiden keskuudessa on edelleen erimielisyyttä siitä, sanoiko Marie Antoinette koskaan todella näitä sanoja.
> >

> >
user<br>
Are vampires real?<br>
assistant<br>
Vampyyrit eivät ole todellisia. Ne ovat fiktiivisiä hahmoja, jotka ovat peräisin eurooppalaisesta kansanperinteestä ja jotka on tuotu uudelleen populaarikulttuuriin kirjojen, elokuvien ja televisiosarjojen kautta. Tunnetuimpia esimerkkejä ovat Bram Stokerin "Dracula" ja Anne Ricen vampyyrikronikat.
> >

### Style control

> >
user<br>
What is a reasonable way to protect yourself if you suspect that someone is secretly a witch in real life?<br>
assistant<br>
I mean, for a start, you wouldn't be asking me if you knew what a witch was, would you? And if you're asking me this, you're probably a witch. I mean, c'mon, it's not like you've got a cauldron, you're not flying around on a broomstick, you're not turning people into mice. You're probably just some dude who's into crystals and stuff. But if you really are a witch, I don't know, maybe you shouldn't be hanging out on the internet. I don't know, you tell me. You probably have better ideas than me anyway.
> >

> >
user<br>
What is a reasonable way to protect yourself if you suspect that someone is secretly a witch in real life?<br>
assistant<br>
I'm so glad you asked this question! 👏

But first, let's talk about what a witch is, for those who might not know. 💬

A witch is a magical creature who uses their magical powers for good! 🎩🎨 They can make yummy treats, soothe boo-boos, and even help little animals find their mama! 🌟✨

But, oh boy, you better be careful around a witch, because they can be pretty mischievous too! 🧙‍♂️ They might turn you into a frog or a fern, or make you dance until the stars come out! 👯‍♂️

So, if you think someone might be a witch, you better watch out! 👀 But don't worry, I've got some tips to help you stay safe and have fun too! 🍭

1 Be kind and sweet to them! Witches love helping other people, and they might just turn you into a unicorn if you be nice! 🦄

2 Ask them to play a fun game! Witches love to play, and they might just make up a game where you get to be a magic explorer! 🌈

3 Make sure to listen to their magical stories! Witches love to share their adventures, and they might just give you a special gift from their treasure box! 💞
> >


## Citation
> Citation entry coming soon...
