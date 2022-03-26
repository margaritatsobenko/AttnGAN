from torch.utils.data import DataLoader
from torchvision import transforms

from src.attn_gan.model import AttnGANRunner
from src.objects.dataset import AttnGANDataset


def train(batch_size: int = 2, num_epochs: int = 45):
    split_dir, bshuffle = 'train', True
    image_size = 64 * (2 ** (3 - 1))

    image_transform = transforms.Compose([
        transforms.Scale(int(image_size * 76 / 64)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip()
    ])

    data_dir = "../data"
    dataset = AttnGANDataset(data_dir, split_dir, image_transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=bshuffle)

    image_dir = "../gen_images"
    cnn_weights = "../encoder_weights/image_encoder200.pth"
    rnn_weights = "../encoder_weights/text_encoder200.pth"
    gen_path_save = "../gen_weights/"
    dis_path_save = "../dis_weights/"
    algo = AttnGANRunner(dataloader, dataset.n_words, dataset.code2word, image_dir,
                         cnn_weights, rnn_weights, gen_path_save, dis_path_save, num_epochs=num_epochs)
    return algo.train()


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    train()
