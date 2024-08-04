<p align="center">
   <img src="https://github.com/AishaEvering/My_Assistant/blob/main/my_assistant_header.png" height="100%" width="700" alt="My Assistant Logo">
</p>

# My Assistant 🤖

In our fast-paced world, having a personal assistant can make life significantly easier. While some are fortunate enough to have one, most of us rely on virtual assistants like Alexa, Siri, Google Assistant, and now, My Assistant.

My Assistant is a fine-tuned DistilBERT model optimized for multiclass text classification. Trained on the [Bhuvaneshwari/intent_classification Hugging Face dataset](https://huggingface.co/datasets/Bhuvaneshwari/intent_classification) from Hugging Face, it accurately predicts customer intents across a range of categories, including: 

[<i>Add To Playlist, Affirmation, Book Meeting, Book Restaurant, Cancellation, Excitment, Get Weather, Greetings, Play Music, Rate Book, Search Creative Work, Search Screening Event</i>]

## Technologies
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

## [Live Demo](https://huggingface.co/spaces/AishaE/My_Assistant)

## 📙 [Jupyter Notebook](https://github.com/AishaEvering/My_Assistant/blob/main/My_Assistant.ipynb)

## Key Takeaways
${{\color{green}\Huge{\textsf{PyTorch\ Seems\ Easier\ with\ Hugging\ Face\ \}}}}\$

During this project, I encountered numerous transformer versioning issues. As a developer accustomed to constant change, I found the rapid evolution in the machine learning field particularly intense. When searching for help online, I often had to check dates, as the community might have moved on from older methods quite quickly. This wasn’t as prominent during my work on a PyTorch project. It might be due to the larger number of developers working with PyTorch or my increased use of Hugging Face transformers. Despite Hugging Face offering both PyTorch and TensorFlow versions of their models, TensorFlow sometimes feels like the “red-headed stepchild”—that’s a reference to Annie, for those who might not be familiar. However, [legacy documentation](https://huggingface.co/transformers/v3.2.0/custom_datasets.html) proved to be incredibly helpful. This experience leads me to my next topic...
***

${{\color{green}\Huge{\textsf{Do\ We\ Still\ Use\ Training\ Arguments?\ \}}}}\$

To address the remaining imbalance in the genre and style labels, I chose to retain the dataset as is and leverage the compute_class_weight function to adjust the class weights in the model's loss function. This approach increased the importance of underrepresented classes while reducing the weight of overrepresented ones. As a result, this adjustment not only reduced the loss but also enhanced the model’s accuracy.
***

${{\color{green}\Huge{\textsf{What\ Is\ BERT?\ \}}}}\$

The dataset exhibited significant imbalance across all labels—artist, genre, and style—with an even greater disparity for the artist label. Over half of the data had unknown artists, and many artists had only a few artworks. Unlike genre and style, the artist label lacked sufficient breadth and diversity for effective model training. Consequently, I decided to remove the artist label to improve the model's overall performance.

In the future, I might revisit this aspect by focusing on a select group of top artists, such as training a model specifically to recognize Vincent Van Gogh's works. For now, the project's focus remains on broadly classifying artworks based on genre and style.

***

${{\color{green}\Huge{\textsf{Final\ Results\ \}}}}\$

In this project, I evaluated the performance of EfficientNet and Vision Transformer (ViT) models using transfer learning. EfficientNet was chosen for its compact size, efficiency, and strong representation of convolutional neural networks (CNNs). Conversely, ViT was selected to explore whether transformers might outperform CNNs for certain tasks. Additionally, I experimented with various image augmentations. However, these experiments were less successful compared to the base models, which is understandable given that altering artwork could significantly change its genre or style.

* **EfficientNet**: This model excelled at predicting the genre of an image but was less effective at predicting the style.
  
   <table>
     <tr>
       <td>
         <strong>Accuracy</strong>
         <ul>
           <li><strong>Genre:</strong> 57%</li>
           <li><strong>Style:</strong> 39%</li>
         </ul>
       </td>
       <td>
         <strong>F1 Score</strong>
         <ul>
           <li><strong>Genre:</strong> 53%
             <ul>
               <li><small><i>Precision:</i> 51% of the predicted genres were correct.</small></li>
               <li><small><i>Recall:</i> The model identified 64% of all actual genres.</small></li>
             </ul>
           </li>
           <li><strong>Style:</strong> 35%
             <ul>
               <li><small><i>Precision:</i> 34% of the predicted styles were correct.</small></li>
               <li><small><i>Recall:</i> The model identified 49% of all actual styles.</small></li>
             </ul>
           </li>
         </ul>
       </td>
     </tr>
   </table>
<br/>
   
* **Vision Transformer**: This model also excelled at predicting the genre of an image and also did better in predicting the style.
  
   <table>
     <tr>
       <td>
         <strong>Accuracy</strong>
         <ul>
           <li><strong>Genre:</strong> 61%</li>
           <li><strong>Style:</strong> 45%</li>
         </ul>
       </td>
       <td>
         <strong>F1 Score</strong>
         <ul>
           <li><strong>Genre:</strong> 58%
             <ul>
               <li><small><i>Precision:</i> 55% of the predicted genres were correct.</small></li>
               <li><small><i>Recall:</i> The model identified 68% of all actual genres.</small></li>
             </ul>
           </li>
           <li><strong>Style:</strong> 43%
             <ul>
               <li><small><i>Precision:</i> 40% of the predicted styles were correct.</small></li>
               <li><small><i>Recall:</i> The model identified 54% of all actual styles.</small></li>
             </ul>
           </li>
         </ul>
       </td>
     </tr>
   </table>

The standout performer is the ViT model, which excels even without additional augmentations. This model is currently the one showcased in the [live demo](https://huggingface.co/spaces/AishaE/art_geek).

## Room for Improvement

* Expand the Dataset: Increasing the amount of data and extending the number of epochs will likely enhance model performance and prediction accuracy.
* Utilize a Learning Rate Scheduler: Implementing a learning rate scheduler can help fine-tune the learning rate, potentially leading to better convergence.
* Focus on Specific Artists: Narrowing the dataset to a select group of artists could provide the model with more focused training, improving its ability to classify art more accurately.
* Incorporate Model Ensembling: Combining predictions from multiple models could lead to more robust and accurate results.

## Summary

This project has been both a challenging and rewarding journey. I had to step back and reassess the metrics, retrain, refactor the code several times, and all of this was done on a borrowed Google GPU via Google Colab, adding to the adventure. Despite the frustrations, I’m incredibly proud of what I’ve accomplished. There’s something truly satisfying about figuring things out as you go. Thank you for following along, and I look forward to sharing my next project. Happy coding!

## Author

Aisha Evering  
[Email](<shovon3000g@gmail.com>) | [Portfolio](https://aishaeportfolio.com/)


