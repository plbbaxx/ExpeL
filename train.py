import getpass
import json
import hydra
from omegaconf import DictConfig
from pathlib import Path
import os 
from copy import deepcopy
from functools import partial
import dotenv
dotenv.load_dotenv()

from agent import AGENT
from prompts.templates.system import system_message_prompt
from prompts.templates.human import HUMAN_CRITIQUES
from prompts import (
    SYSTEM_INSTRUCTION,
    HUMAN_INSTRUCTION,
    FEWSHOTS,
    REFLECTION_FEWSHOTS,
    HUMAN_REFLECTION_INSTRUCTION,
    SYSTEM_REFLECTION_INSTRUCTION,
    SYSTEM_CRITIQUE_INSTRUCTION,
    RULE_TEMPLATE,
    LLM_PARSER,
    OBSERVATION_FORMATTER,
    STEP_IDENTIFIER,
    CYCLER,
    STEP_CYCLER,
    REFLECTION_PREFIX,
    PREVIOUS_TRIALS_FORMATTER,
    STEP_STRIPPER,
    CRITIQUE_SUMMARY_SUFFIX,
)
from envs import ENVS, INIT_TASKS_FN
from memory import (
    EMBEDDERS,
    RETRIEVERS,
)
from models import LLM_CLS
from utils import save_trajectories_log, load_trajectories_log, plot_trial_stats, split_logs_by_task, alfworld_results_per_env_name, get_webshop_mean_scores, get_fewshot_max_tokens, seed_everything
from agent.reflect import Count

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg : DictConfig) -> None:
    seed_everything(int(os.environ.get('EXPEL_SEED', '42')))
    if cfg.testing:
        openai_api_key = 'NO_KEY_FOR_TESTING'
    else:
        openai_api_key = os.environ['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in os.environ else getpass.getpass("Enter or paste your OpenAI API Key: ")
    LOG_PATH = Path('/'.join([cfg.log_dir, cfg.benchmark.name, cfg.agent_type]))
    LOG_PATH.mkdir(parents=True, exist_ok=True)

    # Load trajectory checkpoint, init as empty if not exist
    if cfg.resume:
        out = load_trajectories_log(
        LOG_PATH,
        run_name=cfg.run_name,
        load_true_log=True)
    else:
        # Overwriting confirmation
        if os.path.exists(f"{LOG_PATH}/{cfg.run_name}.pkl") and cfg.run_name != 'test':
            while True:
                res = input(f"Are you sure to overwrite '{cfg.run_name}'? (Y/N)\n").lower()
                if res == 'n':
                    exit(0)
                elif res == 'y':
                    break
        out = {'log': '', 'dicts': [], 'true_log': f'{str(cfg)}'}
    log, dicts, true_log = out['log'], out['dicts'], out['true_log']

    react_agent = AGENT[cfg.agent_type](
        name=cfg.ai_name,
        system_instruction=SYSTEM_INSTRUCTION[cfg.benchmark.name],
        human_instruction=HUMAN_INSTRUCTION[cfg.benchmark.name],
        tasks=INIT_TASKS_FN[cfg.benchmark.name](cfg),
        fewshots=FEWSHOTS[cfg.benchmark.name],
        system_prompt=system_message_prompt,
        env=ENVS[cfg.benchmark.name],
        max_steps=cfg.benchmark.max_steps,
        openai_api_key=openai_api_key,
        llm=cfg.agent.llm,
        llm_builder=LLM_CLS,
        reflection_fewshots=REFLECTION_FEWSHOTS[cfg.benchmark.name],
        reflection_task_prompt=HUMAN_REFLECTION_INSTRUCTION[cfg.benchmark.name],
        reflection_system_instruction=SYSTEM_REFLECTION_INSTRUCTION[cfg.benchmark.name],
        reflection_system_prompt=SYSTEM_INSTRUCTION[cfg.benchmark.name],
        max_relfection_depth=cfg.agent.max_reflection_depth if 'max_reflection_depth' in cfg.agent.keys() else 0,
        system_critique_instructions=SYSTEM_CRITIQUE_INSTRUCTION[cfg.benchmark.name],
        human_critiques=HUMAN_CRITIQUES,
        max_num_rules=cfg.agent.max_num_rules if 'max_num_rules' in cfg.agent.keys() else 0,
        rule_template=RULE_TEMPLATE[cfg.benchmark.name],
        truncate_strategy=cfg.agent.truncate_strategy if 'truncate_strategy' in cfg.agent.keys() else None,
        llm_parser=LLM_PARSER[cfg.benchmark.name],
        observation_formatter=OBSERVATION_FORMATTER[cfg.benchmark.name],
        embedder=EMBEDDERS(cfg.agent.retrieval_kwargs.embedder_type),
        embedder_path=cfg.agent.retrieval_kwargs.embedder_path,
        step_stripper=STEP_STRIPPER[cfg.benchmark.name],
        retriever_cls=RETRIEVERS(cfg.agent.retrieval_kwargs.retriever_type),
        message_splitter=CYCLER[cfg.benchmark.name],
        identifier=STEP_IDENTIFIER[cfg.benchmark.name],
        message_step_splitter=partial(STEP_CYCLER, benchmark=cfg.benchmark.name),
        reflection_prefix=REFLECTION_PREFIX[cfg.benchmark.name],
        previous_trials_formatter=PREVIOUS_TRIALS_FORMATTER[cfg.benchmark.name],
        success_critique_num=cfg.agent.success_critique_num,
        fewshot_strategy=cfg.agent.fewshot_strategy,
        critique_truncate_strategy=cfg.agent.critique_truncate_strategy,
        critique_summary_suffix=CRITIQUE_SUMMARY_SUFFIX,
        testing=cfg.testing,
        task_idx=dicts[-1]['task_idx'] if len(dicts) > 0 else 0,
        benchmark_name=cfg.benchmark.name,
        reranker=cfg.agent.retrieval_kwargs.reranker,
        buffer_retrieve_ratio=cfg.agent.retrieval_kwargs.buffer_retrieve_ratio,
        max_fewshot_tokens=get_fewshot_max_tokens(cfg.benchmark.name) if cfg.agent.retrieval_kwargs.max_fewshot_tokens == 'auto' else cfg.agent.retrieval_kwargs.max_fewshot_tokens,
    )
    if len(dicts) > 0:
        react_agent.load_checkpoint(loaded_dict=dicts[-1], no_load_list=['testing', 'max_relfection_depth', 'fewshot_strategy', 'max_fewshot_tokens'])
        if 'eval_idx_list' in dicts[-1]:
            react_agent.eval_idx_list = dicts[-1]['eval_idx_list']

    print(f"""*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

You are using the following language model: {react_agent.llm.model_name}

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*""")

    while react_agent.job_not_done():
        prefix = f"#######################################\nTASK {react_agent.task_idx}"
        if cfg.agent_type in ['reflection', 'expel']:
            prefix += f' Reflection {react_agent.reflection_counter.count}\n\n'
        else:
            prefix += '\n\n'
        print(prefix + react_agent.remove_task_suffix(react_agent.task)) # remove_task_suffix used for alfworld

        react_agent.run(mode='train')

        #############################################
        ### Update & Save trajectory logs + dicts ###
        #############################################
        react_agent.update_stats()
        log += prefix + react_agent.log_history() + '\n\n'
        true_log += prefix + react_agent.log_history(include_all=True) + '\n\n'
        if cfg.benchmark.name == 'alfworld':
            # Audit only: capture ALFWorld's own admissible-command verdicts
            # without exposing them to the agent or changing its trajectory.
            react_agent.action_audits[str(react_agent.task_idx)] = deepcopy(react_agent.env.action_trace)

        # next task
        react_agent.next_task()
        
        dicts.append({k: deepcopy(v) for k, v in react_agent.__dict__.items() if type(v) in [list, set, str, bool, int, dict, Count] and k not in ['openai_api_key', 'llm']}) # not saving complicated objects
        
        save_trajectories_log(
            LOG_PATH, log, dicts, true_log,
            run_name=cfg.run_name
        )
        #############################################

    ######################################
    ### Final Log & Save stats + PRINT ###
    ######################################
    success, fail, halted = react_agent.get_stats()
    log += f"########################################\nEND TRIAL\nTrial summary: Success: {success}/{success + fail + halted}, Fail: {fail}/{success + fail + halted}, Halted: {halted}/{success + fail + halted}"
    true_log += f"########################################\nEND TRIAL\nTrial summary: Success: {success}/{success + fail + halted}, Fail: {fail}/{success + fail + halted}, Halted: {halted}/{success + fail + halted}"
    print(f'Finished. Success: {success}, Fail: {fail}, Halted: {halted}')

    if cfg.agent_type in ['reflection', 'expel']:
        parsed_result = split_logs_by_task(text=log, num_tasks=len(react_agent.tasks))
        reflection_results = plot_trial_stats(parsed_result=parsed_result, benchmark=cfg.benchmark.name, max_trials=cfg.agent.max_reflection_depth + 1, save_path=f"{LOG_PATH}/{cfg.run_name}_logs_stats.png")
    else:
        # Vanilla React has one terminal outcome per task, rather than
        # reflection trials.  Do not parse free-form text into an inaccurate
        # reflection chart; report the recorded terminal outcomes instead.
        total = success + fail + halted
        reflection_results = {
            'success_rate': round(success / total, 6) if total else 0.0,
            'fail_rate': round(fail / total, 6) if total else 0.0,
            'halted_rate': round(halted / total, 6) if total else 0.0,
        }

    results = ', '.join([f"{k}: {v}" for k, v in reflection_results.items()]) + '\n'
    if cfg.benchmark.name == 'alfworld' and hasattr(react_agent, 'succeeded_trial_history'):
        results += str(alfworld_results_per_env_name(dicts[-1]))
    elif cfg.benchmark.name == 'alfworld':
        # Per-environment success bookkeeping belongs to ExpeL's reflective
        # agent.  Vanilla ReAct has no experience-pool histories, but its
        # overall smoke/pilot summary above remains valid and must be saved.
        results += 'per_environment_results: unavailable_for_vanilla_react\n'
    elif cfg.benchmark.name == 'webshop':
        results += str(get_webshop_mean_scores(log, len(react_agent.tasks), cfg.agent.max_reflection_depth + 1))
    log += f'\n\n{results}\n#######################################'
    true_log += f'\n\n{results}\n#######################################'
    print(results)

    if cfg.benchmark.name == 'alfworld' and hasattr(react_agent, 'action_audits'):
        audit_path = LOG_PATH / f'{cfg.run_name}_action_audit.json'
        audit_path.write_text(json.dumps(react_agent.action_audits, indent=2) + '\n', encoding='utf-8')

    if hasattr(react_agent, 'task_outcomes'):
        total = success + fail + halted
        if total != len(react_agent.tasks):
            raise RuntimeError(
                f'Incomplete terminal accounting: outcomes={total}, tasks={len(react_agent.tasks)}')
        summary_path = LOG_PATH / f'{cfg.run_name}_run_summary.json'
        summary_path.write_text(json.dumps({
            'run_name': cfg.run_name,
            'agent_type': cfg.agent_type,
            'total_tasks': total,
            'success': success,
            'fail': fail,
            'halted': halted,
            'success_rate': success / total if total else 0.0,
            'task_outcomes': react_agent.task_outcomes,
        }, indent=2) + '\n', encoding='utf-8')

    save_trajectories_log(
        LOG_PATH, log, dicts, true_log,
        run_name=cfg.run_name
    )

    log, dicts, true_log = '', [], ''
    react_agent.reset_stats()
    ################################

if __name__ == "__main__":
    main()  
