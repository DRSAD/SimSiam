from torchvision import transforms

class SimSiamTransform():
    def __init__(self, image_size, mean,std):

        p_blur = 0.5 if image_size > 32 else 0

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            #transforms.RandomApply([transforms.GaussianBlur(kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0))], p=p_blur),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
        ])


    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2
