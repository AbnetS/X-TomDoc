<b>Welcome to X-TomDoc</b>

A self-explaining Tomato Disease detection system that is trained to detect healthy and unhealthy tomato leaves attacked by late blight and Septoria leaf spot.

To run the application, follow the steps below:
1. Clone this repository by following the steps below (don't use "git clone" as the repository contains large files)
    > git lfs install
    > git lfs clone https://github.com/AbnetS/X-TomDoc
2. Change the directory to X-TomDoc and install the virtualenv module (This is highly recommended)
    > cd X-TomDoc
    > pip install virtualenv
3. Create a virtual environment (this is recommended)
    > virtualenv env
4. Run the following to activate the newly created environment
    >  source env/scripts/activate
5. Install all dependencies by running the following:
    > pip install -r requirements.txt
6. Run the following to start the application
    > streamlit run 1_😀_\ Welcome.py (recommended to copy and paste this line if typing the emoji is challenging)

<b><u>MODELS</b></u>

You may choose one of the nine models provided with this system. The description of each is given below
1. Vgg16, Vgg19 and IncepV3 => models created by training only the top layer of VGG16, VGG19 and InceptionV3 from Keras libary with the <b>un-segmented</b> version of the PlantVillage dataset.
2. SVgg16, SVgg19 and SIncepV3 => models created by training only the top layer of VGG16, VGG19 and InceptionV3 from Keras libary with the <b>segmented</b> version of the PlantVillage dataset where the backgrond color of all the images is <b>black</b>.
2. SNVgg16, SNVgg19 and SNIncepV3 => models created by training only the top layer of VGG16, VGG19 and InceptionV3 from Keras libary with the <b>segmented</b> version of the PlantVillage dataset where the backgrond color the images is randomly choosen <b>variable color</b>.

<b><u>NOTES</b></u>

The explanations are by default cached. Therefore, choosing a different model but with the same base-model does not invoke the explanation generation and you may obtain the same explanation for a different model(For example, switching between Vgg16 and SVgg16). Click the menu at the very right corner to clear the cache when it is required.

