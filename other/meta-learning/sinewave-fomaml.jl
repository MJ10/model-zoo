using Flux
using Distributions: Uniform
using Plots

# Create a SineWave struct for defining the task.
struct SineWave
    amplitude::Float32
    phase_shift::Float32
end

SineWave() = SineWave(rand(Uniform(0.1, 5)), rand(Uniform(0, 2pi)))

(s::SineWave)(x::AbstractArray) = s.amplitude .* sin.(x .+ s.phase_shift)

# Evaluate model
function base_learning(model, x::AbstractArray, testx::AbstractArray, task=SineWave();
                    opt=Descent(1e-2), updates=32, eval=false)
    weights = params(model)
    prev_weights = deepcopy(Flux.data.(weights))

    y = task(x)
    train_set = zip(x, y)
    testy = task(testx)
    init_preds = model(testx')
    # Make k gradient updates
    for i in 1:updates
        grad = Flux.Tracker.gradient(() -> Flux.mse(model(x'), y'), weights)
        for w in weights
            w.data .-= Flux.Optimise.apply!(opt, w.data, grad[w].data)
        end
        if eval
            test_loss = Flux.mse(model(testx'), testy')
            println("Update ", i, ", Test loss: ", test_loss)
        end
    end

    final_preds = model(testx')

    Flux.loadparams!(model, prev_weights)

    return (x=x, testx=testx, y=y, testy=testy,
            initial_predictions=Array(Flux.data(init_preds)'),
            final_predictions=Array(Flux.data(final_preds)'))
end

# Optimizes model with FOMAML on sampled tasks for better generalization
function fomaml(model; meta_opt=Descent(0.02), inner_opt=Descent(0.02), epochs=30_000,
              n_tasks=3, train_batch_size=10, eval_batch_size=10, eval_interval=10000)

    weights = params(model)
    dist = Uniform(-5, 5)
    testx = Float32.(range(-5, stop=5, length=50))
    for i in 1:epochs
        prev_weights = deepcopy(Flux.data.(weights))

        for _ in 1:n_tasks
            task = SineWave()

            x = Float32.(rand(dist, train_batch_size))
            y = task(x)
            grad = Flux.Tracker.gradient(() -> Flux.mse(model(x'), y'), weights)

            for w in weights
                w.data .-= Flux.Optimise.apply!(inner_opt, w.data, grad[w].data)
            end

            testy = task(testx)
            grad = Flux.Tracker.gradient(() -> Flux.mse(model(testx'), testy'), weights)

            # Reset weights and accumulate gradients
            for (w1, w2) in zip(weights, prev_weights)
                w1.data .= w2
                w1.grad .+= grad[w1].data
            end
        end

        Flux.Optimise._update_params!(meta_opt, weights)

        if i % eval_interval == 0
            # Evaluate the model on a sampled task
            println("Evaluation at epoch ", i)
            evalx = Float32.(rand(dist, eval_batch_size))
            base_learning(model, evalx, testx, SineWave(), eval=true)
        end

    end
end

# Create evaluation task and training data
x = rand(Uniform(-5, 5), 10)
testx = range(-5; stop=5, length=50)
wave = SineWave(4, 1)

# Hyperparameters for training
meta_learning_rate = 0.01
inner_learning_rate = 0.02
n_updates = 32
n_epochs = 50000
n_tasks = 3

fomaml_model = Chain(Dense(1, 64, tanh), Dense(64, 64, tanh), Dense(64, 1))
fomaml(fomaml_model, meta_opt=Descent(meta_learning_rate), inner_opt=Descent(inner_learning_rate), epochs=n_epochs, n_tasks=n_tasks)

result = base_learning(fomaml_model, x, testx, wave, updates=n_updates, opt=Descent(0.02))

plot(
    [result.x, result.testx, result.testx, result.testx],
    [result.y, result.testy, result.initial_predictions, result.final_predictions],
    line=[:scatter :path :path :path],
    label=["Sampled points", "Ground truth", "Before finetune", "After finetune"],
    title="First Order MAML",
    xlim=(-5.5, 5.5)
)
