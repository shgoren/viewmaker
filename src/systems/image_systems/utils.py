from torch.utils.data import DataLoader


def create_dataloader(dataset, config, batch_size, shuffle=True, drop_last=True):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=drop_last,
        num_workers=config.data_loader_workers,
    )
    return loader

def heatmap_of_view_effect(original,view):
    diff = view-original
    diff_offset = (diff.flatten(start_dim=1)-diff.flatten(start_dim=1).min(-1)[0].unsqueeze(-1)).view(diff.shape)
    diff_heatmap = (diff_offset.flatten(start_dim=1)/diff_offset.flatten(start_dim=1).max(-1)[0].unsqueeze(-1)).view(diff_offset.shape)
    return diff_heatmap

def heatmap_of_view_effect_np(original, view):
    diff = view-original
    diff_offset = diff - diff.min((1, 2, 3)).reshape(-1, 1,1,1)
    diff_heatmap = diff_offset / diff_offset.max((1, 2, 3)).reshape(-1, 1,1,1)
    return diff_heatmap