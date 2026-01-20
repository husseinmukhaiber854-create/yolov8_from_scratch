class Trainer:
    """YOLOv8 trainer."""
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None

        # Create directories
        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def setup_data(self):
        """Setup data loaders."""
        print("Setting up datasets...")

        # Download dataset if needed
        if not os.path.exists(self.config.data_dir):
            print("Downloading dataset...")
            CocoDatasetDownloader.download_tiny_coco(self.config.data_dir, num_images=self.config.num_images)

        # Create dataset
        full_dataset = YOLODataset(
            self.config.data_dir,
            img_size=self.config.img_size,
            augment=self.config.use_augmentation
        )

        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=YOLODataset.collate_fn,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=YOLODataset.collate_fn,
            pin_memory=True
        )

        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    def setup_model(self):
        """Setup model, optimizer, and scheduler."""
        print("Setting up model...")

        self.model = YOLOv8(num_classes=self.config.num_classes).to(self.device)

        # Setup optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs
        )

        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            predictions = self.model(images)

            # Calculate loss
            loss_fn = YOLOv8Loss(self.model, self.config.num_classes)
            loss = loss_fn(predictions, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        return total_loss / len(self.train_loader)

    def validate(self):
        """Validate the model."""
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(images)
                loss_fn = YOLOv8Loss(self.model, self.config.num_classes)
                loss = loss_fn(predictions, targets)

                val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint."""
        # Create a pickleable dictionary for configuration parameters
        config_to_save = {key: getattr(self.config, key) for key in dir(self.config) if not key.startswith('_') and not callable(getattr(self.config, key))}

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': config_to_save # Use the pickleable dictionary
        }

        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f'checkpoint_epoch_{epoch}.pth'
        )

        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            shutil.copyfile(checkpoint_path, best_path)

    def plot_results(self, train_losses, val_losses):
        """Plot training results."""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(self.config.save_dir, 'training_loss.png')
        plt.savefig(plot_path)
        plt.close()

    def train(self):
        """Main training loop."""
        print("Starting training...")
        self.setup_data()
        self.setup_model()

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

        for epoch in range(1, self.config.epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.epochs}")

            # Train
            train_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            val_losses.append(val_loss)

            # Update scheduler
            self.scheduler.step()

            # Print progress
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            if epoch % 5 == 0 or epoch == self.config.epochs:
                self.save_checkpoint(epoch, val_loss, is_best)

        # Plot results
        self.plot_results(train_losses, val_losses)

        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")

    def visualize_predictions(self, num_images=3):
        """Visualize model predictions."""
        self.model.eval()

        fig, axes = plt.subplots(num_images, 2, figsize=(15, 5*num_images))
        axes = axes.reshape(-1, 2)

        class_names = ["person", "bicycle", "car", "dog", "cat"]

        with torch.no_grad():
            for idx, (images, targets) in enumerate(self.val_loader):
                if idx >= num_images:
                    break

                images = images.to(self.device)
                predictions = self.model(images)

                # Convert to CPU for visualization
                img_np = images[0].cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)

                # Plot original with ground truth
                ax1 = axes[idx, 0]
                ax1.imshow(img_np)
                ax1.set_title("Ground Truth")

                # Plot predictions
                ax2 = axes[idx, 1]
                ax2.imshow(img_np)
                ax2.set_title("Predictions")

                # Draw ground truth boxes
                target = targets[0]
                for t in target:
                    if t[1] != 0:  # Skip padded entries
                        cls_id = int(t[1])
                        x_center, y_center, width, height = t[2:6].numpy()

                        # Convert from normalized to pixel coordinates
                        h, w = img_np.shape[:2]
                        x1 = int((x_center - width/2) * w)
                        y1 = int((y_center - height/2) * h)
                        x2 = int((x_center + width/2) * w)
                        y2 = int((x_center + height/2) * h)

                        rect = Rectangle((x1, y1), x2-x1, y2-y1,
                                       linewidth=2, edgecolor='green', facecolor='none')
                        ax1.add_patch(rect)
                        ax1.text(x1, y1-5, class_names[cls_id],
                               color='green', fontsize=10,
                               bbox=dict(facecolor='white', alpha=0.7))

                plt.tight_layout()

        vis_path = os.path.join(self.config.save_dir, 'predictions.png')
        plt.savefig(vis_path)
        plt.close()
        print(f"Visualizations saved to {vis_path}")
        
def main():
    """Main training function."""
    # Print configuration
    Config.print_config()

    # Create trainer
    trainer = Trainer(Config)

    # Start training
    start_time = time.time()
    trainer.train()
    end_time = time.time()

    print(f"Total training time: {(end_time - start_time)/60:.2f} minutes")

    # Visualize predictions
    trainer.visualize_predictions()