import torchvision
from torch.utils.tensorboard import SummaryWriter
import os

class TensorboardVisualizer():
    def __init__(self,log_dir='./logs/'):
        self.log_dir = log_dir

    def prepare_folder(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def visualize_image_batch(self,image_batch,n_iter,image_name='Image_batch'):
        grid = torchvision.utils.make_grid(image_batch)
        self.writer.add_image(image_name,grid,n_iter)

    def plot_loss(self, loss_val, n_iter, loss_name='loss'):
        self.writer.add_scalar(loss_name, loss_val, n_iter)

    def plot_metrics(self, value, n_iter, metric_name):
        self.writer.add_scalar(metric_name, value, n_iter)

    def plot_projection(self, emb, image, meta):
        self.writer.add_embedding(emb.view(emb.shape[0], -1), metadata=meta, label_img=image)

    def plot_graph(self, model, images):
        self.writer.add_graph(model, images)
        
    def close(self):
        self.writer.close()