"""Shared JSON schemas for team execution tools."""

from nanobot.agent.tools.schema import ArraySchema, ObjectSchema, StringSchema


def team_tasks_schema() -> ArraySchema:
    return ArraySchema(
        items=ObjectSchema(
            id=StringSchema("Unique task id.", min_length=1),
            role=StringSchema("Worker role, such as researcher or reviewer.", min_length=1),
            instruction=StringSchema("Focused instruction for this worker.", min_length=1),
            dependsOn=ArraySchema(
                StringSchema("Task id that must complete first."),
                description="Task ids whose successful results are required.",
            ),
            capabilityProfile=StringSchema(
                "Built-in capability profile controlling the worker tool set.",
                enum=["general", "implement", "research", "review"],
            ),
            required=["id", "role", "instruction"],
        ),
        description="Explicit task graph to execute.",
        min_items=1,
        max_items=32,
    )
