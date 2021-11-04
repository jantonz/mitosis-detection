from fastai.vision.all import *

# from fastai.vision import *
# from fastai.vision.data import ImageDataLoaders
# from fastai.data.external import Path
# from fastai.metrics import error_rate, accuracy

class OverSamplingCallback(Callback):
    def __init__(self,learn:Learner):
        super().__init__(learn)
        self.labels = self.learn.data.train_dl.dataset.y.items
        _, counts = np.unique(self.labels,return_counts=True)
        self.weights = torch.DoubleTensor((1/counts)[self.labels])
        self.label_counts = np.bincount([self.learn.data.train_dl.dataset.y[i].data for i in range(len(self.learn.data.train_dl.dataset))])
        self.total_len_oversample = int(self.learn.data.c*np.max(self.label_counts))
        
    def on_train_begin(self, **kwargs):
        self.learn.data.train_dl.dl.batch_sampler = BatchSampler(WeightedRandomSampler(weights,self.total_len_oversample), self.learn.data.train_dl.batch_size,False)


# Load data ####
# Set path to root directory
path = Path("../dataset")

# View all files in directory
path.ls()

# # Data augmentation
# path_hr = path / "train" / '1'
# il = ImageDataLoaders.from_folder(path)
# tfms = aug_transforms(max_rotate=25)


# def data_aug_one(ex_img, prox, qnt):
#     for lop in range(0, qnt):
#         image_name = str(prox).zfill(8) + ".jpg"
#         dest = path_hr / image_name
#         prox = prox + 1
#         new_img = open_image(ex_img)
#         new_img_fin = new_img.apply_tfms(
#             tfms[0], new_img, xtra={tfms[1][0].tfm: {"size": 224}}, size=224
#         )
#         new_img_fin.save(dest)


# prox = 20
# qnt = 10
# for imagen in il.items:
#     data_aug_one(imagen, prox, qnt)
#     prox = prox + qnt


# We are creating a fastai DataBunch from our dataset
data = ImageDataLoaders.from_folder(
    path,
    train="train",
    valid="test",
    # ds_tfms=aug_transforms(do_flip=False),
    size=224,
    bs=64,
    num_workers=4,
)

# Show what the data looks like after being transformed
data.show_batch()
# See the number of images in each data set
print(len(data.train_ds), len(data.valid_ds))

# Train model ####
# Build the CNN model with the pretrained resnet18
learn = cnn_learner(
    data,
    resnet34,
    # pretrained=True,
    # loss_func=CrossEntropyLossFlat(),
    metrics=[accuracy],
    model_dir="../model/",
)

learn.fine_tune(epochs=1)

lr = learn.lr_find()

learn.fine_tune(4, 1e-2)

learn.show_results(max_n=4)



# Train the model on 4 epochs of data at the default learning rate
learn.fit_one_cycle(4)

# Save the model
learn.save("stage-1")

# Load the Model
learn.load("stage-1")

# Unfreeze all layers of the CNN
learn.unfreeze()

# Find the optimal learning rate and plot a visual
learn.lr_find()

# Fit the model over 2 epochs
learn.fit_one_cycle(2, max_lr=slice(3e-7, 5e-7))

# Explore results ####
# Rebuild interpreter and replot confusion matrix
interp = ClassificationInterpretation.from_learner(learn)

# Confusion matrix
interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)

# Worst predictions (Predicted/Actual/Sum)
interp.most_confused()
interp.top_losses()

# Show images of worst predictions
interp.plot_top_losses(8)
