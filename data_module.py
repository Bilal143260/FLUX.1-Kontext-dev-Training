from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image

class KontextDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        dataset_name,
        source_column_name,
        target_column_name,
        caption_column_name,
        size=(512, 512),
        max_samples=None,
    ):
        self.dataset_name = dataset_name
        self.source_column_name = source_column_name
        self.target_column_name = target_column_name
        self.caption_column_name = caption_column_name
        self.size = size

        self.buckets = buckets
        # Load the dataset from HuggingFace Hub
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "You are trying to load your data using the datasets library. Please install it via `pip install datasets`."
            )
        self.dataset = load_dataset(self.dataset_name)
        # Limit to a maximum of max_samples samples if available
        if max_samples is not None and max_samples < len(self.dataset["train"]):
            self.dataset["train"] = self.dataset["train"].select(range(max_samples))
        self._length = len(self.dataset["train"])
        
        # Setup transformations
        width, height = self.size
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        # Load images on demand from the dataset
        sample = self.dataset["train"][index]
        source_image = sample[self.source_column_name]
        target_image = sample[self.target_column_name]
        
        # Convert image modes if needed
        if source_image.mode != "RGB":
            source_image = source_image.convert("RGB")
        if target_image.mode != "RGB":
            target_image = target_image.convert("RGB")
        
        # Apply transformations
        source_tensor = self.train_transforms(source_image)
        target_tensor = self.train_transforms(target_image)
        raw_caption = sample[self.caption_column_name]
        
        # Return the concatenated image, mask, and caption in a dictionary
        example = {
            "target_image": target_tensor,
            "source_image": source_tensor,
            "caption": raw_caption,
        }
        
        return example


def collate_fn(batch):
    """
    Custom collate function for BgAiDataset to properly batch the concatenated images, masks, and captions.
    
    Args:
        batch: List of samples from BgAiDataset.
        
    Returns:
        Dictionary with batched concatenated images, masks, and a list of captions.
    """
    source_images = torch.stack([item['source_image'] for item in batch])
    target_images = torch.stack([item['target_image'] for item in batch])
    captions = [item['caption'] for item in batch]
    
    
    return {
        "source_image": source_images, #conditioned on this 
        "target_image": target_images, # target_image (noise to be added on it)
        "caption": captions,
    }

if __name__ == "__main__":
    # Create dataset instance with the updated parameters:
    dataset = KontextDataset(
        dataset_name="raresense/Viton",
        target_column_name="source",
        source_column_name="target",
        caption_column_name="ai_name",
    )
    
    # Print dataset length
    print(f"Dataset size: {len(dataset)}")
    
    # Get first sample
    print("Loading first sample...")
    sample = dataset[0]
    
    # Print tensor shapes
    print("\nTensor shape of concatenated image:")
    print(f"Target image: {sample['target_image'].shape}")
    print(f"Mask: {sample['mask'].shape}")
    
    # Print caption (truncated if too long)
    caption = sample["caption"]
    if len(caption) > 100:
        caption = caption[:100] + "..."
    print(f"\nCaption: {caption}")

    # Create and test DataLoader with custom collate function
    batch_size = 4
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    print("\nTesting batch processing:")
    print(f"Getting first batch of size {batch_size}...")
    batch = next(iter(dataloader))
    
    # Print tensor shapes for batch
    print("Tensor shapes for batch:")
    print(f"Target images: {batch['target_image'].shape}")
    print(f"Masks: {batch['mask'].shape}")
    print(f"Sources: {batch['source_image'].shape}")
    print(f"Number of captions: {len(batch['caption'])}")
    
    # Print first caption in batch
    first_caption = batch['caption'][0]
    if len(first_caption) > 100:
        first_caption = first_caption
    print(f"First caption in batch: {first_caption}")