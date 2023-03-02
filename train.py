
#delete some images 20% from all dataset
input_train = input_train[:100000,:,:]
input_test = input_test[:9000,:,:]
target_train = target_train[:100000]
target_test  = target_test[:9000]

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np

# Model configuration
batch_size = 64
img_width, img_height, img_num_channels = 32, 32, 3
loss_function = sparse_categorical_crossentropy
no_classes = 9
no_epochs = 30
#optimizer = Adam()
adam = Adam(lr=0.0005,beta_1=0.9, beta_2=0.999, epsilon=1e-06)
validation_split = 0.2
verbosity = 1
num_folds = 5

from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)

# Parse numbers as floats
input_train = input_train.astype('float32')

input_test = input_test.astype('float32')

# Normalize data
input_train = input_train / 255
input_test = input_test / 255

target_train = tf.keras.utils.to_categorical(target_train )
target_test = tf.keras.utils.to_categorical(target_test )

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []
prec_per_fold= []
recall_per_fold= []
f1_per_fold=[]
##############################################
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)



###############################################

# Merge inputs and targets
inputs = np.concatenate((input_train, input_test))
targets = np.concatenate((target_train, target_test))
############################################################
histories=list()

###########################################################
# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):
  
   model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy',precision,recall, f1])
   print('------------------------------------------------------------------------')
   print(f'Training for fold {fold_no} ...')

   history = model.fit(inputs[train], targets[train],
              batch_size=batch_size,
              epochs=no_epochs,
              verbose=verbosity,
              validation_data=(input_test, target_test))
   histories.append(history)         
  
  # Generate generalization metrics
   scores = model.evaluate(inputs[test], targets[test], verbose=0)
   print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%; {model.metrics_names[2]} of {scores[2]*100}%; {model.metrics_names[3]} of {scores[3]*100}%; {model.metrics_names[4]} of {scores[4]*100}%')
   acc_per_fold.append(scores[1] * 100)
   prec_per_fold.append(scores[2] * 100)
   recall_per_fold.append(scores[3] * 100)
   f1_per_fold.append(scores[4] * 100)
   loss_per_fold.append(scores[0])

  # Increase fold number
   fold_no = fold_no + 1
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%- Precision: {prec_per_fold[i]}%- Recall: {recall_per_fold[i]}%- F1_Score: {f1_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print(f'> Precision: {np.mean(prec_per_fold)} (+- {np.std(prec_per_fold)})')
print(f'> Recall: {np.mean(recall_per_fold)} (+- {np.std(recall_per_fold)})')
print(f'> F1_Score: {np.mean(f1_per_fold)} (+- {np.std(f1_per_fold)})')
print('------------------------------------------------------------------------')
######################################################################


# plot diagnostic learning curves
fig , ax = plt.subplots(1,2)
fig.set_size_inches(6,10)
colours = ['b','g','r','y','k']
labels_Train   = ['Fold1_Train','Fold2_Train', 'Fold3_Train', 'Fold4_Train', 'Fold5_Train']
labels_Test  = ['Fold1_Test', 'Fold2_Test','Fold3_Test','Fold4_Test','Fold5_Test' ]
for i in range(len(histories)):
		# plot loss
		pyplot.subplot(211)
		pyplot.title('Cross Entropy Loss')
		pyplot.plot(histories[i].history['loss'], color=colours[i], label=labels_Train[i])
		pyplot.plot(histories[i].history['val_loss'], color=colours[i], label=labels_Test[i], linestyle = "--")
pyplot.legend(loc='upper left') 
pyplot.plot()

  		# plot accuracy

for i in range(len(histories)):      
		pyplot.subplot(212)
		pyplot.title('Classification Accuracy')
		pyplot.plot(histories[i].history['accuracy'],  color=colours[i], label=labels_Train[i])
		pyplot.plot(histories[i].history['val_accuracy'], color=colours[i], label=labels_Test[i], linestyle = "--")
pyplot.legend(loc='upper left') 
pyplot.plot()
 
#######################################################
pred = model.predict(inputs[test])
# ROC curve
fpr, tpr, thresholds = metrics.roc_curve(targets[test].argmax(axis=1), pred.argmax(axis=1))
roc_auc = metrics.auc(fpr, tpr)
roc_display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

roc_display.plot()
plt.title('ROC_CURVE')
roc_display.figure_.savefig(f'/content/sample_data{fold_no}.jpeg')
.3
# PR Curve
precision, recall, _ = metrics.precision_recall_curve(targets[test].argmax(axis=1), pred.argmax(axis=1))
pr_display = metrics.PrecisionRecallDisplay(precision=precision, recall=recall)
pr_display.plot()
plt.title('PR_Curve')
pr_display.figure_.savefig(f'/content/PR_curve_for_fold#{fold_no}.jpeg')



