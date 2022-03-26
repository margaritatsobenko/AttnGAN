import torch


def prepare_data(batch, device):
    images, captions, captions_len, class_ids, file_names = batch

    sorted_cap_lens, sorted_cap_indices = torch.sort(captions_len, 0, True)

    sorted_images = []
    for i in range(len(images)):
        sorted_images.append(images[i][sorted_cap_indices].to(device))

    sorted_captions = captions[sorted_cap_indices].squeeze().to(device)
    sorted_cap_lens = sorted_cap_lens.to(device)
    sorted_file_names = [file_names[i] for i in sorted_cap_indices.numpy()]
    class_ids = class_ids[sorted_cap_indices].numpy()

    return sorted_images, sorted_captions, sorted_cap_lens, class_ids, sorted_file_names
