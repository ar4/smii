def backpropagate(propagator, image, stored_source_wavefield):

    for step in reversed(range(1, propagator.timestep.num_steps)):
        source_wavefield = stored_source_wavefield.restore(step-1)
        receiver_wavefield = propagator.step()
        image.add(source_wavefield, receiver_wavefield)

    return image.image
