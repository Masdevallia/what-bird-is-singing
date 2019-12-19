
<p align="center"><img  src="https://raw.githubusercontent.com/Masdevallia/what-bird-is-singing/master/images/logo.jpg" width="400"></p>

## <p align="center">What bird is singing?</p>
<p align="center">http://masdevallia.pythonanywhere.com/</p>

### <p align="center">Ironhack's Data Analytics Bootcamp Final Project</p>

## 

<p align="center"><img  src="https://raw.githubusercontent.com/Masdevallia/what-bird-is-singing/master/images/photosall.png" width="900"></p>

<p align="center"><img  src="https://raw.githubusercontent.com/Masdevallia/what-bird-is-singing/master/images/index.jpg" width="700"></p>

### STEP 1: Idea generation
Bringing technology closer to biologists needs.

### STEP 2: Obtaining the database
The database (consistent of 14509 audios for 317 species) was obtained from the [Xeno-Canto API](https://www.xeno-canto.org/explore/api).

### STEP 3: Audio processing
- Low pass filter.
- Noise gata.
- Split on silence: separate out silent chunks.
- Split each remaining chunk into 1 second windows overlapping by 50%.
- Calculate Fourier coefficients for each window.
- Determine the average value of each mfcc coefficient for each window.

### STEP 4: Model building (training and evaluation)
Tryed: Sklearn classifiers (random forest, gradient boosting, ...) and neural networks.

Best performance: Convolutional neural network with 2D convolutional layers and MaxPooling2D (validation accuracy = 0.95).

### STEP 5: Web application and deployment
- Developed with Flask .
- Deployed online with Pythonanywhere.

### Future steps:
- 'Others' category.
- Add more species.
- Improve the neural network architecture.