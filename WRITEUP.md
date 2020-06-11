# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves:

Adding extensions to both the Model Optimizer and the Inference Engine.

Some of the potential reasons for handling custom layers are:

Due to the fact that openvino supports several frameworks like (TensorFlow, Caffe, MXNet..),
The toolkit Openvino has a list of supported layers from these frameworks.
if a model uses a layer which is not inluded in this list, it will be considered as a custom layer and must be reworked as mentionned earlier.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:
Statistcs about customers in a shop.
Like for pandamic COVID-19 only a certain number is allowed in a shop so this app can be used to help the owners of the shop to control the flow of the people

Each of these use cases would be useful because
it reduce the human effort and remove the boring jobs

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

Lighting : The more the lightning the better will be the accuracy of the detection.

Model accuracy : The accuracy is one of the most important if not the most important parameter in this project. using the other model listed bellow the accuracy was not that good so all the calculation were not correct so I couldn't proceed with these models

Camera focal length : helps a lot with the detection of persons.

Image size : affects directly the time of the image processing. the bigger the size is the more time will be spent to proccess the image. 

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: SSD MobileNet V2 COCO
  - https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html
  - I converted the model to an Intermediate Representation with the following arguments : python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  - The model was insufficient for the app because it doesn't detect the second guy in the video
  - I tried to improve the model for the app by lowering the prob_threshold and processing less frames 
  
- Model 2: SSD ResNet50 FPN COCO
  - https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html
  - I converted the model to an Intermediate Representation with the following arguments : python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config
  - The model was insufficient for the app because the video flow was too slow, frames are not coming fast.
  - I tried to improve the model for the app by : Iwas not able to impove it

- Model 3: SSD Lite MobileNet V2 COCO
  - https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html
  - I converted the model to an Intermediate Representation with the following arguments : python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config
  - The model was insufficient for the app because the accuracy was not good at all
  - I tried to improve the model for the app by lowering the prob_threshold and processing less frames 

- In the end I used the pretrained model person-detection-retail-0013 and it worked perfectly.