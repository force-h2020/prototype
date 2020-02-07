# fourFn.py
#
# Demonstration of the pyparsing module, implementing a simple 4-function expression parser,
# with support for scientific notation, and symbols for e and pi.
# Extended to add exponentiation and simple built-in functions.
# Extended test cases, simplified pushFirst method.
# Removed unnecessary expr.suppress() call (thanks Nathaniel Peterson!), and added Group
# Changed fnumber to use a Regex, which is now the preferred method
#
# Copyright 2003-2009 by Paul McGuire
#
from pyparsing import Literal,Word,Group,\
    ZeroOrMore,Forward,alphas,alphanums,Regex,ParseException,\
    CaselessKeyword, Suppress
import math
import operator

global exprStack
exprStack = []

def pushFirst( strg, loc, toks ):
    exprStack.append( toks[0] )
def pushUMinus( strg, loc, toks ):
    for t in toks:
      if t == '-':
        exprStack.append( 'unary -' )
        #~ exprStack.append( '-1' )
        #~ exprStack.append( '*' )
      else:
        break

bnf = None
def BNF():
    """
    expop   :: '^'
    multop  :: '*' | '/'
    addop   :: '+' | '-'
    integer :: ['+' | '-'] '0'..'9'+
    atom    :: PI | E | real | fn '(' expr ')' | '(' expr ')'
    factor  :: atom [ expop factor ]*
    term    :: factor [ multop factor ]*
    expr    :: term [ addop term ]*
    """
    global bnf
    if not bnf:
        point = Literal( "." )
        # use CaselessKeyword for e and pi, to avoid accidentally matching
        # functions that start with 'e' or 'pi' (such as 'exp'); Keyword
        # and CaselessKeyword only match whole words
        e     = CaselessKeyword( "E" )
        pi    = CaselessKeyword( "PI" )
        #~ fnumber = Combine( Word( "+-"+nums, nums ) +
                           #~ Optional( point + Optional( Word( nums ) ) ) +
                           #~ Optional( e + Word( "+-"+nums, nums ) ) )
        fnumber = Regex(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
        ident = Word(alphas, alphanums+"_$")

        plus, minus, mult, div = map(Literal, "+-*/")
        lpar, rpar = map(Suppress, "()")
        addop  = plus | minus
        multop = mult | div
        expop = Literal( "^" )

        expr = Forward()
        atom = ((0,None)*minus + ( pi | e | fnumber | ident + lpar + expr + rpar | ident ).setParseAction( pushFirst ) |
                Group( lpar + expr + rpar )).setParseAction(pushUMinus)

        # by defining exponentiation as "atom [ ^ factor ]..." instead of "atom [ ^ atom ]...", we get right-to-left exponents, instead of left-to-righ
        # that is, 2^3^2 = 2^(3^2), not (2^3)^2.
        factor = Forward()
        factor << atom + ZeroOrMore( ( expop + factor ).setParseAction( pushFirst ) )

        term = factor + ZeroOrMore( ( multop + factor ).setParseAction( pushFirst ) )
        expr << term + ZeroOrMore( ( addop + term ).setParseAction( pushFirst ) )
        bnf = expr
    return bnf

# map operator symbols to corresponding arithmetic operations
epsilon = 1e-12
opn = { "+" : operator.add,
        "-" : operator.sub,
        "*" : operator.mul,
        "/" : operator.truediv,
        "^" : operator.pow}
fn  = { "sin" : math.sin,
        "cos" : math.cos,
        "tan" : math.tan,
        "exp" : math.exp,
        "abs" : abs,
        "trunc" : lambda a: int(a),
        "round" : round,
        "sgn" : lambda a: (a > epsilon) - (a < -epsilon),
        "ln" : math.log,
        "log2": math.log2,
        "log10": math.log10 }

def evaluateStack(s, var_set, func_set):
    op = s.pop()
    if op == 'unary -':
        return -evaluateStack( s, var_set, func_set)
    if op in "+-*/^":
        op2 = evaluateStack( s, var_set, func_set)
        op1 = evaluateStack( s, var_set, func_set)
        return opn[op]( op1, op2 )
    elif op == "PI":
        return math.pi # 3.1415926535
    elif op == "E":
        return math.e  # 2.718281828
    elif op in fn:
        return fn[op]( evaluateStack( s, var_set, func_set) )
    elif op in func_set:
        return evaluateStack(func_set[op][:], var_set, func_set)
    elif op in var_set:
        return var_set[op]
    elif op == "invalid":
        raise Exception("error in dependency")
    elif op[0].isalpha():
        raise Exception("invalid identifier '%s'" % op)
    else:
        return float( op )

def evaluateFunction( s, var_set, func_set ):
    #var_set and func_set must be disjoint
    try:
        val = evaluateStack( s[:], var_set, func_set)
    except ParseException as pe:
        print(s, "failed parse:", str(pe))
    except Exception as e:
        print(s, "failed eval:", str(e))
    else:
        return val

def parseFunction( s ):
    exprStack[:] = []
    try:
        BNF().parseString( s, parseAll=True)
    except ParseException as pe:
        print(s, "failed parse:", str(pe))
        raise
    else:
        return exprStack[:]