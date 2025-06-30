from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
import torch
from PIL import Image

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        class_prompt,
        class_data_root=None,
        class_num=None,
        repeats=1,
        center_crop=False,
        buckets=None,
    ):
        self.center_crop = center_crop

        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt

        self.buckets = buckets

        # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
        # we load the training data using load_dataset
        if args.dataset_name is not None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "You are trying to load your data using the datasets library. If you wish to train using custom "
                    "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                    "local folder containing images only, specify --instance_data_dir instead."
                )
            # Downloading and loading a dataset from the hub.
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )
            # Preprocessing the datasets.
            column_names = dataset["train"].column_names

            # 6. Get the column names for input/target.
            if args.image_column is None:
                image_column = column_names[0]
                logger.info(f"image column defaulting to {image_column}")
            else:
                image_column = args.image_column
                if image_column not in column_names:
                    raise ValueError(
                        f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
            instance_images = dataset["train"][image_column]

            if args.caption_column is None:
                logger.info(
                    "No caption column provided, defaulting to instance_prompt for all images. If your dataset "
                    "contains captions/prompts for the images, make sure to specify the "
                    "column as --caption_column"
                )
                self.custom_instance_prompts = None
            else:
                if args.caption_column not in column_names:
                    raise ValueError(
                        f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
                custom_instance_prompts = dataset["train"][args.caption_column]
                # create final list of captions according to --repeats
                self.custom_instance_prompts = []
                for caption in custom_instance_prompts:
                    self.custom_instance_prompts.extend(itertools.repeat(caption, repeats))
        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exists.")

            instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir())]
            self.custom_instance_prompts = None

        self.instance_images = []
        for img in instance_images:
            self.instance_images.extend(itertools.repeat(img, repeats))

        self.pixel_values = []
        for image in self.instance_images:
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")

            width, height = image.size

            # Find the closest bucket
            bucket_idx = find_nearest_bucket(height, width, self.buckets)
            target_height, target_width = self.buckets[bucket_idx]
            self.size = (target_height, target_width)

            # based on the bucket assignment, define the transformations
            train_resize = transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR)
            train_crop = transforms.CenterCrop(self.size) if center_crop else transforms.RandomCrop(self.size)
            train_flip = transforms.RandomHorizontalFlip(p=1.0)
            train_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            image = train_resize(image)
            if args.center_crop:
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, self.size)
                image = crop(image, y1, x1, h, w)
            if args.random_flip and random.random() < 0.5:
                image = train_flip(image)
            image = train_transforms(image)
            self.pixel_values.append((image, bucket_idx))

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.size) if center_crop else transforms.RandomCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image, bucket_idx = self.pixel_values[index % self.num_instance_images]
        example["instance_images"] = instance_image
        example["bucket_idx"] = bucket_idx

        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            if caption:
                example["instance_prompt"] = caption
            else:
                example["instance_prompt"] = self.instance_prompt

        else:  # custom prompts were provided, but length does not match size of image dataset
            example["instance_prompt"] = self.instance_prompt

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt"] = self.class_prompt

        return example


def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts}
    return batch

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

if __name__ == "__main__":
    # Create dataset instance with the updated parameters:
    dataset = PosetoFlatlayDataset(
        dataset_name="raresense/Viton",
        source_column_name="target",
        target_column_name="source",
        caption_column_name="ai_name",
        mask_local_path="mask_square.png",  # Local mask image file
        # max_samples defaults to 5000, limiting the dataset to 5,000 samples.
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