from subprocess import call, run
import json

message = "first10x10run"
obstacles_sources = [f"TEST0{i}" for i in range(10)]
agents_sources = [f"TEST0{i}" for i in range(10)]
models = ["6-test-iters30", "65-test-iters30", "7-test-iters30"]

for model in models:

    model_results = {
        'obstacles': [],
        'agents': [], 
        'avg_alignm_after_100': [],
        'avg_cohesi_after_100': [],
        'avg_diamet_after_100': [],
        'avg_collis_after_100': [],
        'avg_connec_after_100': [],
        'avg_phipxy_after_100': [],
        'all_phipxy': []
    }
    
    for obstacles in obstacles_sources:
        for agents in agents_sources:
            # [model_name, obstacles, agents, destination, message, gif_flag, plot_flag]
            # call(["python3", "evaluation.py"] + [model, obstacles, agents, "", message, "", ""])
            result = run(["python3", "evaluation.py"] + [model, obstacles, agents, "", message, "" if agents != "TEST00" else "gif", "" if agents != "TEST00" else "plot"], capture_output=True, text=True)
            stdout = result.stdout
            start = stdout.find("{")
            end = stdout.rfind("}") + 1
            output = json.loads(stdout[start:end])

            for key in model_results:
                model_results[key].append(output[key])

    with open(f"metrics/{message}_{model}.json", "w") as f:
        json.dump(model_results, f, indent=4)
