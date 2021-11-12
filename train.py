# next_time_pred pkgs
old_target=df[['target']]
df=df.drop(['target'],axis=1)


# Number of training samples.
sample_count = len(train_sample)
epochs = 50

# Number of warmup epochs.
warmup_epoch = 10

# Base learning rate after warmup.
learning_rate_base = 0.001

total_steps = int(epochs * sample_count / batch_size)

# Compute the number of warmup batches.
warmup_steps = int(warmup_epoch * sample_count / batch_size)

warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                        total_steps=total_steps,
                                        warmup_learning_rate=0.0,
                                        warmup_steps=warmup_steps,
                                        hold_base_rate_steps=0)
