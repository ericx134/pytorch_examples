import torch
import torch.nn as nn
import torch.nn.functional as F

0. Tensor operations:

torch.Tensor.repeat(num_dim0, num_dim1, num_dim2, ...) #repeat the given tensor num_dim* times along each dimension provided
torch.Tensor.transpose(dim0, dim1) #swap the two dimensions of the tensor.


1. if use F.dropout() inside a nn.Module class, need to pass self.training to differentiate training vs testing
If use class nn.Dropout() inside nn.Modele, then self.training is automatically passed in.

example:

Class DropoutFC(nn.Module):
  def __init__(self):
      super(DropoutFC, self).__init__()
      self.fc = nn.Linear(100,20)
      self.dropout = nn.Dropout(p=0.5) #no self.training needed, implicitly passed 

  def forward(self, input):
      out = self.fc(input)
      out = self.dropout(out)
      return out


class DropoutFC(nn.Module):
   def __init__(self):
       super(DropoutFC, self).__init__()
       self.fc = nn.Linear(100,20)

   def forward(self, input):
       out = self.fc(input)
       out = F.dropout(out, p=0.5, training=self.training) #explicit passing self.training is needed.
       return out



2. tensor.max(), if not used globally like below, return value are two tensors (max_value, argmax_index)
>>>a = torch.randn(2, 2)
>>>a.max()


3. python multi-processing

def init_processes(rank, size, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []

    start = time.time()
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run, 'gloo'))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end = time.time()
    print "Elapsed time: %.2f" % (end - start)


4. Common layers with important parameters 
nn.Linear(m, n)

Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
>>> layer = torch.nn.ConvTranspose2d(3, 64, 3, 1, 0)

BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
>>> layer = torch.nn.BatchNorm2d(64)

ReLU(inplace=False)
LeakyReLU(0.2, inplace=True)
>>> layer = torch.nn.ReLU(True)




5. Initialize weights for nn.Module with nn.Module.apply(fn)
# custom weights initialization called on netG and netD
def weights_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        module.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)
module = Net()
module.apply(weights_init)



6. cuda related calls and attributes
- cuda = torch.device('cuda')     # Default CUDA device
  cuda0 = torch.device('cuda:0')
  cuda2 = torch.device('cuda:2')  # GPU 2 (these are 0-indexed)

- x = torch.tensor([1, 2], device=cuda)

- with torch.cuda.device(1):
    a = torch.tensor([1., 2.], device=cuda) #a.device is device(type='cuda', index=1

- operations on cuda tensors, result will be on the same cuda device; cross-device operations not allowed.

- torch.tensor.is_cuda
- device = torch.device("cuda:0" if opt.cuda else "cpu")
- if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

- fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)




import argparse
import torch

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
device = None
if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

x = torch.empty((8, 42), device=args.device)
module = Net().to(device=device)


cuda0 = torch.device('cuda:0')  # CUDA GPU 0
for i, x in enumerate(train_loader):
    x = x.to(cuda0)




- set device in multi-gpu environment 
torch.cuda.set_device(3)
a = torch.tensor([1, 2]).cuda #a.device is device(type='cuda', index=3) / "cuda:3"






7. Serialization (saving modules)

#save
torch.save(the_model.state_dict(), PATH)
#load
the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))

#saving checkpoint 1
if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                transformer.eval().cpu() #make the model no grad, then convert back to cpu. Saving state dict requires the model convert back to the cpu.
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train() #convert back to gpu or whatever device it was on, then enter training mode.

#save checkpoint 2
if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval ==0:
    with torch.no_grad():
      transformer.cpu() #still need to convert to cpu
      ckpt_model_filename = "ckpt_epock_" + str(e) + "_batch_id_" + str(batch_id+1) + ".pth"
      ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
      torch.save(transformer.state_dict(), ckpt_model_path)
      transformer.to(device)


#example from imagenet example
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

# saved object can also be a self defined dictionary
save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
# optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




8. Image Dataset Loader:

torchvision.datasets.ImageFolder(root, transform=None, target_transform=None, loader=<function default_loader>)
>>> import torchvision.datasets as datasets
>>> import torchvision as transforms
>>> transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))  
    ])
>>> dataset = datasets.ImageFolder(dataset_path, transform)
# A generic data loader where the images are arranged in this way:

# root/dog/xxx.png
# root/dog/xxy.png
# root/dog/xxz.png

# root/cat/123.png
# root/cat/nsdf3.png
# root/cat/asd932_.png



style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
style = utils.load_image(args.style_image, size=args.style_size)
#Compose can be applied on images directly
style = style_transform(style)
style = style.repeat(args.batch_size, 1, 1, 1).to(device)





Create custom dataset:
1. torch.utils.data.Dataset is an abstract class to build custom dataset upon
2. Inherent torch.utils.data.Dataset
3. Do initializing in __init__()
4. Overload __len__() and __getitem__() to support length query and indexing into dataset.

from torch.utils.data import Dataset

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

9. Using pretrained model (part of it):


#VGG
from collections import namedtuple

import torch
from torchvision import models


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


torch.nn.Module.add_module(name, module)
