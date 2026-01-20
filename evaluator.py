import gradio as gr
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv

# Ensure evaluation modules are in path or installed
from evaluation.eval import evaluate_all_retrieval, evaluate_all_answers
from evaluation.eval_canonical import evaluate_scifact

load_dotenv(override=True)

# Color coding thresholds
MRR_GREEN, MRR_AMBER = 0.9, 0.75
NDCG_GREEN, NDCG_AMBER = 0.9, 0.75
COVERAGE_GREEN, COVERAGE_AMBER = 90.0, 75.0
ANSWER_GREEN, ANSWER_AMBER = 4.5, 4.0


def run_canonical_retrieval_evaluation(use_reranker, progress=gr.Progress()):
    """Runs SciFact canonical retrieval evaluation."""
    progress(0, desc="Starting SciFact evaluation...")

    # Pass use_reranker to the evaluation function
    results = evaluate_scifact(use_reranker=use_reranker)

    avg_mrr_at_k = results["avg_mrr_at_k"]
    avg_ndcg_at_k = results["avg_ndcg_at_k"]
    overall_mrr = results["overall_mrr"]
    overall_ndcg = results["overall_ndcg"]

    html_blocks = ""
    html_blocks += format_metric_html("Overall MRR", overall_mrr, "mrr")
    html_blocks += format_metric_html("Overall nDCG", overall_ndcg, "ndcg")

    for k in sorted(avg_mrr_at_k.keys()):
        html_blocks += format_metric_html(f"MRR@{k}", avg_mrr_at_k[k], "mrr")
        html_blocks += format_metric_html(f"nDCG@{k}", avg_ndcg_at_k[k], "ndcg")

    df = pd.DataFrame(
        {
            "K": list(avg_mrr_at_k.keys()),
            "MRR": list(avg_mrr_at_k.values()),
            "nDCG": list(avg_ndcg_at_k.values()),
        }
    )

    return html_blocks, df


def get_color(value: float, metric_type: str) -> str:
    """Get color based on metric value and type."""
    if metric_type == "mrr":
        return (
            "green" if value >= MRR_GREEN else "orange" if value >= MRR_AMBER else "red"
        )
    elif metric_type == "ndcg":
        return (
            "green"
            if value >= NDCG_GREEN
            else "orange" if value >= NDCG_AMBER else "red"
        )
    elif metric_type == "coverage":
        return (
            "green"
            if value >= COVERAGE_GREEN
            else "orange" if value >= COVERAGE_AMBER else "red"
        )
    elif metric_type in ["accuracy", "completeness", "relevance"]:
        return (
            "green"
            if value >= ANSWER_GREEN
            else "orange" if value >= ANSWER_AMBER else "red"
        )
    return "black"


def format_metric_html(
    label, value, metric_type, is_percentage=False, score_format=False
):
    """Format a metric with color coding."""
    color = get_color(value, metric_type)
    if is_percentage:
        value_str = f"{value:.1f}%"
    elif score_format:
        value_str = f"{value:.2f}/5"
    else:
        value_str = f"{value:.4f}"
    return f"""
    <div style="margin: 10px 0; padding: 15px; background-color: #f5f5f5; border-radius: 8px; border-left: 5px solid {color};">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">{label}</div>
        <div style="font-size: 28px; font-weight: bold; color: {color};">{value_str}</div>
    </div>
    """


def run_retrieval_evaluation(use_reranker, progress=gr.Progress()):
    """Run retrieval evaluation (LLM tests)."""
    total_mrr = 0.0
    total_ndcg = 0.0
    total_coverage = 0.0
    category_mrr = defaultdict(list)
    count = 0

    # Pass use_reranker to the generator
    for test, result, prog_value in evaluate_all_retrieval(use_reranker=use_reranker):
        count += 1
        total_mrr += result.mrr
        total_ndcg += result.ndcg
        total_coverage += result.keyword_coverage
        category_mrr[test.category].append(result.mrr)
        progress(prog_value, desc=f"Evaluating test {count}...")

    avg_mrr = total_mrr / count if count else 0
    avg_ndcg = total_ndcg / count if count else 0
    avg_coverage = total_coverage / count if count else 0

    final_html = f"""
    <div style="padding: 0;">
        {format_metric_html("Mean Reciprocal Rank (MRR)", avg_mrr, "mrr")}
        {format_metric_html("Normalized DCG (nDCG)", avg_ndcg, "ndcg")}
        {format_metric_html("Keyword Coverage", avg_coverage, "coverage", is_percentage=True)}
        <div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
            <span style="font-size: 14px; color: #155724; font-weight: bold;">✓ Evaluation Complete: {count} tests</span>
        </div>
    </div>
    """

    category_data = [
        {"Category": k, "Average MRR": sum(v) / len(v)} for k, v in category_mrr.items()
    ]
    df = pd.DataFrame(category_data)
    return final_html, df


def run_answer_evaluation(use_reranker, progress=gr.Progress()):
    """Run answer evaluation."""
    total_accuracy = 0.0
    total_completeness = 0.0
    total_relevance = 0.0
    category_accuracy = defaultdict(list)
    count = 0

    # Pass use_reranker to the generator
    for test, result, prog_value in evaluate_all_answers(use_reranker=use_reranker):
        count += 1
        total_accuracy += result.accuracy
        total_completeness += result.completeness
        total_relevance += result.relevance
        category_accuracy[test.category].append(result.accuracy)
        progress(prog_value, desc=f"Evaluating test {count}...")

    avg_accuracy = total_accuracy / count if count else 0
    avg_completeness = total_completeness / count if count else 0
    avg_relevance = total_relevance / count if count else 0

    final_html = f"""
    <div style="padding: 0;">
        {format_metric_html("Accuracy", avg_accuracy, "accuracy", score_format=True)}
        {format_metric_html("Completeness", avg_completeness, "completeness", score_format=True)}
        {format_metric_html("Relevance", avg_relevance, "relevance", score_format=True)}
        <div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
            <span style="font-size: 14px; color: #155724; font-weight: bold;">✓ Evaluation Complete: {count} tests</span>
        </div>
    </div>
    """

    category_data = [
        {"Category": k, "Average Accuracy": sum(v) / len(v)}
        for k, v in category_accuracy.items()
    ]
    df = pd.DataFrame(category_data)
    return final_html, df


def main():
    """Launch the Gradio evaluation app."""
    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="RAG Evaluation Dashboard", theme=theme) as app:
        gr.Markdown("# 📊 RAG Evaluation Dashboard")
        gr.Markdown("Evaluate retrieval and answer quality for the Scifact RAG system")

        # --- NEW: RERANKER TOGGLE ---
        with gr.Row():
            reranker_checkbox = gr.Checkbox(
                label="Enable Reranker (LLM-based)",
                value=False,
                info="If checked, documents will be reranked by the LLM before evaluation. This increases time and API cost.",
            )

        gr.Markdown("## 📈 SciFact Canonical Retrieval Evaluation")
        canonical_button = gr.Button(
            "Run SciFact Retrieval Eval", variant="primary", size="lg"
        )
        with gr.Row():
            with gr.Column(scale=1):
                canonical_metrics = gr.HTML(
                    "<div style='padding:20px;text-align:center;color:#999;'>Click to start</div>"
                )
            with gr.Column(scale=1):
                canonical_chart = gr.BarPlot(
                    x="K",
                    y="MRR",
                    y2="nDCG",
                    title="MRR & nDCG by K",
                    y_lim=[0, 1],
                    height=400,
                )

        gr.Markdown("## 🔍 Retrieval Evaluation (LLM generated test set)")
        retrieval_button = gr.Button("Run Evaluation", variant="primary", size="lg")
        with gr.Row():
            with gr.Column(scale=1):
                retrieval_metrics = gr.HTML(
                    "<div style='padding: 20px; text-align: center; color: #999;'>Click 'Run Evaluation' to start</div>"
                )
            with gr.Column(scale=1):
                retrieval_chart = gr.BarPlot(
                    x="Category",
                    y="Average MRR",
                    title="Average MRR by Category",
                    y_lim=[0, 1],
                    height=400,
                )

        gr.Markdown("## 💬 Answer Evaluation (GPT-5-Nano as a Judge)")
        answer_button = gr.Button("Run Evaluation", variant="primary", size="lg")
        with gr.Row():
            with gr.Column(scale=1):
                answer_metrics = gr.HTML(
                    "<div style='padding: 20px; text-align: center; color: #999;'>Click 'Run Evaluation' to start</div>"
                )
            with gr.Column(scale=1):
                answer_chart = gr.BarPlot(
                    x="Category",
                    y="Average Accuracy",
                    title="Average Accuracy by Category",
                    y_lim=[1, 5],
                    height=400,
                )

        # Wire up the evaluations with the checkbox input
        canonical_button.click(
            fn=run_canonical_retrieval_evaluation,
            inputs=[reranker_checkbox],
            outputs=[canonical_metrics, canonical_chart],
        )

        retrieval_button.click(
            fn=run_retrieval_evaluation,
            inputs=[reranker_checkbox],
            outputs=[retrieval_metrics, retrieval_chart],
        )

        answer_button.click(
            fn=run_answer_evaluation,
            inputs=[reranker_checkbox],
            outputs=[answer_metrics, answer_chart],
        )

    app.launch(inbrowser=True)


if __name__ == "__main__":
    main()
