import matplotlib.pyplot as plt


# todo change a lot this one
def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).

    predictions = model(test_input, training=False)

    #  Plot the generated images
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    # 3 - Save the generated images
    plt.savefig(f'images_generated/image_epoch_{epoch}.png')
    plt.show()
