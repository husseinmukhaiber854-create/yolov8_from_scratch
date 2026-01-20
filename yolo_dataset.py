class YOLODataset(Dataset):
    def __init__(self, data_dir, img_size=640, augment=False):
        self.data_dir = data_dir
        self.img_size = img_size
        self.augment = augment
        self.image_dir = os.path.join(data_dir, "images")
        self.label_dir = os.path.join(data_dir, "labels")

        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(glob.glob(os.path.join(self.image_dir, ext)))

        # Define augmentations
        self.transform = self.get_augmentations() if augment else self.get_base_transform()

    def get_base_transform(self):
        """Base transformation without augmentation."""
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.1
        ))

    def get_augmentations(self):
        """Augmentation pipeline."""
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.HorizontalFlip(p=Config.flip_prob),
            A.HueSaturationValue(
                hue_shift_limit=int(Config.hsv_h * 180),
                sat_shift_limit=int(Config.hsv_s * 255),
                val_shift_limit=int(Config.hsv_v * 255),
                p=0.5
            ),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.1),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.MotionBlur(blur_limit=3, p=0.1),
            ], p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.1
        ))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load labels
        label_path = os.path.join(
            self.label_dir,
            os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        )

        bboxes = []
        class_labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)

        # Apply transformations
        if len(bboxes) > 0:
            transformed = self.transform(
                image=img,
                bboxes=bboxes,
                class_labels=class_labels
            )
            img = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        else:
            transformed = self.transform(image=img)
            img = transformed['image']
            bboxes = []
            class_labels = []

        # Convert to tensor format
        labels = torch.zeros((len(bboxes), 6))
        for i, (bbox, cls_id) in enumerate(zip(bboxes, class_labels)):
            labels[i, 0] = 0  # batch index (will be set in collate_fn)
            labels[i, 1] = cls_id
            labels[i, 2:] = torch.tensor(bbox)

        return img, labels

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for YOLO dataset."""
        images, labels = zip(*batch)
        images = torch.stack(images, 0)

        # Update batch indices in labels
        max_labels = max(len(l) for l in labels)
        padded_labels = []
        for batch_idx, label_set in enumerate(labels):
            if len(label_set) > 0:
                label_set[:, 0] = batch_idx
            padded = torch.zeros((max_labels, 6))
            if len(label_set) > 0:
                padded[:len(label_set)] = label_set
            padded_labels.append(padded)

        labels = torch.stack(padded_labels, 0)
        return images, labels