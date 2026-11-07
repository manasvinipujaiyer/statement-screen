"""Synthetic bank-statement generation (all data is fake)."""

from statementscreen.synth.generator import generate_statement
from statementscreen.synth.models import IncomeTruth, Statement, Transaction

__all__ = ["IncomeTruth", "Statement", "Transaction", "generate_statement"]
