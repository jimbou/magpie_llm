
You have been assigned the role of crossover assistant during genetic programming. You are assisting in the crossover between multiple parents. The goal is to create a child program that is a combination of the multiple parents. The parents are represented as a series of edits. The source file (code program or parameter file)  you are working on is presented below to give you context.
Your task is to select from the available parents the edits you think are the more beneficial to create a child program. The child program must be a combination of the edits from the available parents. The child must be a combination of the available edits. Your response must adhere to the text format: Child: ***the child***.

The source file:
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<unit xmlns="http://www.srcML.org/srcML/src" xmlns:cpp="http://www.srcML.org/srcML/cpp" revision="0.9.5" language="C++" filename="examples/code/minisat/core/Solver.cc"><comment type="block" format="doxygen">/***************************************************************************************[Solver.cc]
Copyright (c) 2003-2006, Niklas Een, Niklas Sorensson
Copyright (c) 2007-2010, Niklas Sorensson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
**************************************************************************************************/</comment>

<cpp:include>#<cpp:directive>include</cpp:directive> <cpp:file>&lt;math.h&gt;</cpp:file></cpp:include>

<cpp:include>#<cpp:directive>include</cpp:directive> <cpp:file>"mtl/Sort.h"</cpp:file></cpp:include>
<cpp:include>#<cpp:directive>include</cpp:directive> <cpp:file>"core/Solver.h"</cpp:file></cpp:include>

<using>using <namespace>namespace <name>Minisat</name>;</namespace></using>

<comment type="line">//=================================================================================================</comment>
<comment type="line">// Options:</comment>


<decl_stmt><decl><specifier>static</specifier> <type><specifier>const</specifier> <name>char</name><modifier>*</modifier></type> <name>_cat</name> <init>= <expr><literal type="string">"CORE"</literal></expr></init></decl>;</decl_stmt>

<decl_stmt><decl><specifier>static</specifier> <type><name>DoubleOption</name></type>  <name>opt_var_decay</name>         <argument_list>(<argument><expr><name>_cat</name></expr></argument>, <argument><expr><literal type="string">"var-decay"</literal></expr></argument>,   <argument><expr><literal type="string">"The variable activity decay factor"</literal></expr></argument>,            <argument><expr><literal type="number">0.95</literal></expr></argument>,     <argument><expr><call><name>DoubleRange</name><argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>, <argument><expr><literal type="boolean">false</literal></expr></argument>, <argument><expr><literal type="number">1</literal></expr></argument>, <argument><expr><literal type="boolean">false</literal></expr></argument>)</argument_list></call></expr></argument>)</argument_list></decl>;</decl_stmt>
<decl_stmt><decl><specifier>static</specifier> <type><name>DoubleOption</name></type>  <name>opt_clause_decay</name>      <argument_list>(<argument><expr><name>_cat</name></expr></argument>, <argument><expr><literal type="string">"cla-decay"</literal></expr></argument>,   <argument><expr><literal type="string">"The clause activity decay factor"</literal></expr></argument>,              <argument><expr><literal type="number">0.999</literal></expr></argument>,    <argument><expr><call><name>DoubleRange</name><argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>, <argument><expr><literal type="boolean">false</literal></expr></argument>, <argument><expr><literal type="number">1</literal></expr></argument>, <argument><expr><literal type="boolean">false</literal></expr></argument>)</argument_list></call></expr></argument>)</argument_list></decl>;</decl_stmt>
<decl_stmt><decl><specifier>static</specifier> <type><name>DoubleOption</name></type>  <name>opt_random_var_freq</name>   <argument_list>(<argument><expr><name>_cat</name></expr></argument>, <argument><expr><literal type="string">"rnd-freq"</literal></expr></argument>,    <argument><expr><literal type="string">"The frequency with which the decision heuristic tries to choose a random variable"</literal></expr></argument>, <argument><expr><literal type="number">0</literal></expr></argument>, <argument><expr><call><name>DoubleRange</name><argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>, <argument><expr><literal type="boolean">true</literal></expr></argument>, <argument><expr><literal type="number">1</literal></expr></argument>, <argument><expr><literal type="boolean">true</literal></expr></argument>)</argument_list></call></expr></argument>)</argument_list></decl>;</decl_stmt>
<decl_stmt><decl><specifier>static</specifier> <type><name>DoubleOption</name></type>  <name>opt_random_seed</name>       <argument_list>(<argument><expr><name>_cat</name></expr></argument>, <argument><expr><literal type="string">"rnd-seed"</literal></expr></argument>,    <argument><expr><literal type="string">"Used by the random variable selection"</literal></expr></argument>,         <argument><expr><literal type="number">91648253</literal></expr></argument>, <argument><expr><call><name>DoubleRange</name><argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>, <argument><expr><literal type="boolean">false</literal></expr></argument>, <argument><expr><name>HUGE_VAL</name></expr></argument>, <argument><expr><literal type="boolean">false</literal></expr></argument>)</argument_list></call></expr></argument>)</argument_list></decl>;</decl_stmt>
<decl_stmt><decl><specifier>static</specifier> <type><name>IntOption</name></type>     <name>opt_ccmin_mode</name>        <argument_list>(<argument><expr><name>_cat</name></expr></argument>, <argument><expr><literal type="string">"ccmin-mode"</literal></expr></argument>,  <argument><expr><literal type="string">"Controls conflict clause minimization (0=none, 1=basic, 2=deep)"</literal></expr></argument>, <argument><expr><literal type="number">2</literal></expr></argument>, <argument><expr><call><name>IntRange</name><argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>, <argument><expr><literal type="number">2</literal></expr></argument>)</argument_list></call></expr></argument>)</argument_list></decl>;</decl_stmt>
<decl_stmt><decl><specifier>static</specifier> <type><name>IntOption</name></type>     <name>opt_phase_saving</name>      <argument_list>(<argument><expr><name>_cat</name></expr></argument>, <argument><expr><literal type="string">"phase-saving"</literal></expr></argument>, <argument><expr><literal type="string">"Controls the level of phase saving (0=none, 1=limited, 2=full)"</literal></expr></argument>, <argument><expr><literal type="number">2</literal></expr></argument>, <argument><expr><call><name>IntRange</name><argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>, <argument><expr><literal type="number">2</literal></expr></argument>)</argument_list></call></expr></argument>)</argument_list></decl>;</decl_stmt>
<decl_stmt><decl><specifier>static</specifier> <type><name>BoolOption</name></type>    <name>opt_rnd_init_act</name>      <argument_list>(<argument><expr><name>_cat</name></expr></argument>, <argument><expr><literal type="string">"rnd-init"</literal></expr></argument>,    <argument><expr><literal type="string">"Randomize the initial activity"</literal></expr></argument>, <argument><expr><literal type="boolean">false</literal></expr></argument>)</argument_list></decl>;</decl_stmt>
<decl_stmt><decl><specifier>static</specifier> <type><name>BoolOption</name></type>    <name>opt_luby_restart</name>      <argument_list>(<argument><expr><name>_cat</name></expr></argument>, <argument><expr><literal type="string">"luby"</literal></expr></argument>,        <argument><expr><literal type="string">"Use the Luby restart sequence"</literal></expr></argument>, <argument><expr><literal type="boolean">true</literal></expr></argument>)</argument_list></decl>;</decl_stmt>
<decl_stmt><decl><specifier>static</specifier> <type><name>IntOption</name></type>     <name>opt_restart_first</name>     <argument_list>(<argument><expr><name>_cat</name></expr></argument>, <argument><expr><literal type="string">"rfirst"</literal></expr></argument>,      <argument><expr><literal type="string">"The base restart interval"</literal></expr></argument>, <argument><expr><literal type="number">100</literal></expr></argument>, <argument><expr><call><name>IntRange</name><argument_list>(<argument><expr><literal type="number">1</literal></expr></argument>, <argument><expr><name>INT32_MAX</name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></decl>;</decl_stmt>
<decl_stmt><decl><specifier>static</specifier> <type><name>DoubleOption</name></type>  <name>opt_restart_inc</name>       <argument_list>(<argument><expr><name>_cat</name></expr></argument>, <argument><expr><literal type="string">"rinc"</literal></expr></argument>,        <argument><expr><literal type="string">"Restart interval increase factor"</literal></expr></argument>, <argument><expr><literal type="number">2</literal></expr></argument>, <argument><expr><call><name>DoubleRange</name><argument_list>(<argument><expr><literal type="number">1</literal></expr></argument>, <argument><expr><literal type="boolean">false</literal></expr></argument>, <argument><expr><name>HUGE_VAL</name></expr></argument>, <argument><expr><literal type="boolean">false</literal></expr></argument>)</argument_list></call></expr></argument>)</argument_list></decl>;</decl_stmt>
<decl_stmt><decl><specifier>static</specifier> <type><name>DoubleOption</name></type>  <name>opt_garbage_frac</name>      <argument_list>(<argument><expr><name>_cat</name></expr></argument>, <argument><expr><literal type="string">"gc-frac"</literal></expr></argument>,     <argument><expr><literal type="string">"The fraction of wasted memory allowed before a garbage collection is triggered"</literal></expr></argument>,  <argument><expr><literal type="number">0.20</literal></expr></argument>, <argument><expr><call><name>DoubleRange</name><argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>, <argument><expr><literal type="boolean">false</literal></expr></argument>, <argument><expr><name>HUGE_VAL</name></expr></argument>, <argument><expr><literal type="boolean">false</literal></expr></argument>)</argument_list></call></expr></argument>)</argument_list></decl>;</decl_stmt>


<comment type="line">//=================================================================================================</comment>
<comment type="line">// Constructor/Destructor:</comment>


<constructor><name><name>Solver</name><operator>::</operator><name>Solver</name></name><parameter_list>()</parameter_list> <member_init_list>:

    <comment type="line">// Parameters (user settable):</comment>
    <comment type="line">//</comment>
    <call><name>verbosity</name>        <argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>)</argument_list></call>
  , <call><name>var_decay</name>        <argument_list>(<argument><expr><name>opt_var_decay</name></expr></argument>)</argument_list></call>
  , <call><name>clause_decay</name>     <argument_list>(<argument><expr><name>opt_clause_decay</name></expr></argument>)</argument_list></call>
  , <call><name>random_var_freq</name>  <argument_list>(<argument><expr><name>opt_random_var_freq</name></expr></argument>)</argument_list></call>
  , <call><name>random_seed</name>      <argument_list>(<argument><expr><name>opt_random_seed</name></expr></argument>)</argument_list></call>
  , <call><name>luby_restart</name>     <argument_list>(<argument><expr><name>opt_luby_restart</name></expr></argument>)</argument_list></call>
  , <call><name>ccmin_mode</name>       <argument_list>(<argument><expr><name>opt_ccmin_mode</name></expr></argument>)</argument_list></call>
  , <call><name>phase_saving</name>     <argument_list>(<argument><expr><name>opt_phase_saving</name></expr></argument>)</argument_list></call>
  , <call><name>rnd_pol</name>          <argument_list>(<argument><expr><literal type="boolean">false</literal></expr></argument>)</argument_list></call>
  , <call><name>rnd_init_act</name>     <argument_list>(<argument><expr><name>opt_rnd_init_act</name></expr></argument>)</argument_list></call>
  , <call><name>garbage_frac</name>     <argument_list>(<argument><expr><name>opt_garbage_frac</name></expr></argument>)</argument_list></call>
  , <call><name>restart_first</name>    <argument_list>(<argument><expr><name>opt_restart_first</name></expr></argument>)</argument_list></call>
  , <call><name>restart_inc</name>      <argument_list>(<argument><expr><name>opt_restart_inc</name></expr></argument>)</argument_list></call>

    <comment type="line">// Parameters (the rest):</comment>
    <comment type="line">//</comment>
  , <call><name>learntsize_factor</name><argument_list>(<argument><expr><operator>(</operator><name>double</name><operator>)</operator><literal type="number">1</literal><operator>/</operator><operator>(</operator><name>double</name><operator>)</operator><literal type="number">3</literal></expr></argument>)</argument_list></call>, <call><name>learntsize_inc</name><argument_list>(<argument><expr><literal type="number">1.1</literal></expr></argument>)</argument_list></call>

    <comment type="line">// Parameters (experimental):</comment>
    <comment type="line">//</comment>
  , <call><name>learntsize_adjust_start_confl</name> <argument_list>(<argument><expr><literal type="number">100</literal></expr></argument>)</argument_list></call>
  , <call><name>learntsize_adjust_inc</name>         <argument_list>(<argument><expr><literal type="number">1.5</literal></expr></argument>)</argument_list></call>

    <comment type="line">// Statistics: (formerly in 'SolverStats')</comment>
    <comment type="line">//</comment>
  , <call><name>solves</name><argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>)</argument_list></call>, <call><name>starts</name><argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>)</argument_list></call>, <call><name>decisions</name><argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>)</argument_list></call>, <call><name>rnd_decisions</name><argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>)</argument_list></call>, <call><name>propagations</name><argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>)</argument_list></call>, <call><name>conflicts</name><argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>)</argument_list></call>
  , <call><name>dec_vars</name><argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>)</argument_list></call>, <call><name>clauses_literals</name><argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>)</argument_list></call>, <call><name>learnts_literals</name><argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>)</argument_list></call>, <call><name>max_literals</name><argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>)</argument_list></call>, <call><name>tot_literals</name><argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>)</argument_list></call>

  , <call><name>ok</name>                 <argument_list>(<argument><expr><literal type="boolean">true</literal></expr></argument>)</argument_list></call>
  , <call><name>cla_inc</name>            <argument_list>(<argument><expr><literal type="number">1</literal></expr></argument>)</argument_list></call>
  , <call><name>var_inc</name>            <argument_list>(<argument><expr><literal type="number">1</literal></expr></argument>)</argument_list></call>
  , <call><name>watches</name>            <argument_list>(<argument><expr><call><name>WatcherDeleted</name><argument_list>(<argument><expr><name>ca</name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call>
  , <call><name>qhead</name>              <argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>)</argument_list></call>
  , <call><name>simpDB_assigns</name>     <argument_list>(<argument><expr><operator>-</operator><literal type="number">1</literal></expr></argument>)</argument_list></call>
  , <call><name>simpDB_props</name>       <argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>)</argument_list></call>
  , <call><name>order_heap</name>         <argument_list>(<argument><expr><call><name>VarOrderLt</name><argument_list>(<argument><expr><name>activity</name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call>
  , <call><name>progress_estimate</name>  <argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>)</argument_list></call>
  , <call><name>remove_satisfied</name>   <argument_list>(<argument><expr><literal type="boolean">true</literal></expr></argument>)</argument_list></call>

    <comment type="line">// Resource constraints:</comment>
    <comment type="line">//</comment>
  , <call><name>conflict_budget</name>    <argument_list>(<argument><expr><operator>-</operator><literal type="number">1</literal></expr></argument>)</argument_list></call>
  , <call><name>propagation_budget</name> <argument_list>(<argument><expr><operator>-</operator><literal type="number">1</literal></expr></argument>)</argument_list></call>
  , <call><name>asynch_interrupt</name>   <argument_list>(<argument><expr><literal type="boolean">false</literal></expr></argument>)</argument_list></call>
</member_init_list><block>{}</block></constructor>


<destructor><name><name>Solver</name><operator>::</operator>~<name>Solver</name></name><parameter_list>()</parameter_list>
<block>{
}</block></destructor>


<comment type="line">//=================================================================================================</comment>
<comment type="line">// Minor methods:</comment>


<comment type="line">// Creates a new SAT variable in the solver. If 'decision' is cleared, variable will not be</comment>
<comment type="line">// used as a decision variable (NOTE! This has effects on the meaning of a SATISFIABLE result).</comment>
<comment type="line">//</comment>
<function><type><name>Var</name></type> <name><name>Solver</name><operator>::</operator><name>newVar</name></name><parameter_list>(<parameter><decl><type><name>bool</name></type> <name>sign</name></decl></parameter>, <parameter><decl><type><name>bool</name></type> <name>dvar</name></decl></parameter>)</parameter_list>
<block>{
    <decl_stmt><decl><type><name>int</name></type> <name>v</name> <init>= <expr><call><name>nVars</name><argument_list>()</argument_list></call></expr></init></decl>;</decl_stmt>
    <expr_stmt><expr><call><name><name>watches</name>  <operator>.</operator><name>init</name></name><argument_list>(<argument><expr><call><name>mkLit</name><argument_list>(<argument><expr><name>v</name></expr></argument>, <argument><expr><literal type="boolean">false</literal></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <expr_stmt><expr><call><name><name>watches</name>  <operator>.</operator><name>init</name></name><argument_list>(<argument><expr><call><name>mkLit</name><argument_list>(<argument><expr><name>v</name></expr></argument>, <argument><expr><literal type="boolean">true</literal></expr></argument> )</argument_list></call></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <expr_stmt><expr><call><name><name>assigns</name>  <operator>.</operator><name>push</name></name><argument_list>(<argument><expr><name>l_Undef</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <expr_stmt><expr><call><name><name>vardata</name>  <operator>.</operator><name>push</name></name><argument_list>(<argument><expr><call><name>mkVarData</name><argument_list>(<argument><expr><name>CRef_Undef</name></expr></argument>, <argument><expr><literal type="number">0</literal></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <comment type="line">//activity .push(0);</comment>
    <expr_stmt><expr><call><name><name>activity</name> <operator>.</operator><name>push</name></name><argument_list>(<argument><expr><ternary><condition><expr><name>rnd_init_act</name></expr> ?</condition><then> <expr><call><name>drand</name><argument_list>(<argument><expr><name>random_seed</name></expr></argument>)</argument_list></call> <operator>*</operator> <literal type="number">0.00001</literal></expr> </then><else>: <expr><literal type="number">0</literal></expr></else></ternary></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <expr_stmt><expr><call><name><name>seen</name>     <operator>.</operator><name>push</name></name><argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <expr_stmt><expr><call><name><name>polarity</name> <operator>.</operator><name>push</name></name><argument_list>(<argument><expr><name>sign</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <expr_stmt><expr><call><name><name>decision</name> <operator>.</operator><name>push</name></name><argument_list>()</argument_list></call></expr>;</expr_stmt>
    <expr_stmt><expr><call><name><name>trail</name>    <operator>.</operator><name>capacity</name></name><argument_list>(<argument><expr><name>v</name><operator>+</operator><literal type="number">1</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <expr_stmt><expr><call><name>setDecisionVar</name><argument_list>(<argument><expr><name>v</name></expr></argument>, <argument><expr><name>dvar</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <return>return <expr><name>v</name></expr>;</return>
}</block></function>


<function><type><name>bool</name></type> <name><name>Solver</name><operator>::</operator><name>addClause_</name></name><parameter_list>(<parameter><decl><type><name><name>vec</name><argument_list type="generic">&lt;<argument><expr><name>Lit</name></expr></argument>&gt;</argument_list></name><modifier>&amp;</modifier></type> <name>ps</name></decl></parameter>)</parameter_list>
<block>{
    <expr_stmt><expr><call><name>assert</name><argument_list>(<argument><expr><call><name>decisionLevel</name><argument_list>()</argument_list></call> <operator>==</operator> <literal type="number">0</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <if>if <condition>(<expr><operator>!</operator><name>ok</name></expr>)</condition><then> <block type="pseudo"><return>return <expr><literal type="boolean">false</literal></expr>;</return></block></then></if>

    <comment type="line">// Check if clause is satisfied and remove false/duplicate literals:</comment>
    <expr_stmt><expr><call><name>sort</name><argument_list>(<argument><expr><name>ps</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <decl_stmt><decl><type><name>Lit</name></type> <name>p</name></decl>;</decl_stmt> <decl_stmt><decl><type><name>int</name></type> <name>i</name></decl>, <decl><type ref="prev"/><name>j</name></decl>;</decl_stmt>
    <for>for <control>(<init><expr><name>i</name> <operator>=</operator> <name>j</name> <operator>=</operator> <literal type="number">0</literal></expr><operator>,</operator> <expr><name>p</name> <operator>=</operator> <name>lit_Undef</name></expr>;</init> <condition><expr><name>i</name> <operator>&lt;</operator> <call><name><name>ps</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>i</name><operator>++</operator></expr></incr>)</control>
        <block type="pseudo"><if>if <condition>(<expr><call><name>value</name><argument_list>(<argument><expr><name><name>ps</name><index>[<expr><name>i</name></expr>]</index></name></expr></argument>)</argument_list></call> <operator>==</operator> <name>l_True</name> <operator>||</operator> <name><name>ps</name><index>[<expr><name>i</name></expr>]</index></name> <operator>==</operator> <operator>~</operator><name>p</name></expr>)</condition><then>
            <block type="pseudo"><return>return <expr><literal type="boolean">true</literal></expr>;</return></block></then>
        <elseif>else <if>if <condition>(<expr><call><name>value</name><argument_list>(<argument><expr><name><name>ps</name><index>[<expr><name>i</name></expr>]</index></name></expr></argument>)</argument_list></call> <operator>!=</operator> <name>l_False</name> <operator>&amp;&amp;</operator> <name><name>ps</name><index>[<expr><name>i</name></expr>]</index></name> <operator>!=</operator> <name>p</name></expr>)</condition><then>
            <block type="pseudo"><expr_stmt><expr><name><name>ps</name><index>[<expr><name>j</name><operator>++</operator></expr>]</index></name> <operator>=</operator> <name>p</name> <operator>=</operator> <name><name>ps</name><index>[<expr><name>i</name></expr>]</index></name></expr>;</expr_stmt></block></then></if></elseif></if></block></for>
    <expr_stmt><expr><call><name><name>ps</name><operator>.</operator><name>shrink</name></name><argument_list>(<argument><expr><name>i</name> <operator>-</operator> <name>j</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>

    <if>if <condition>(<expr><call><name><name>ps</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call> <operator>==</operator> <literal type="number">0</literal></expr>)</condition><then>
        <block type="pseudo"><return>return <expr><name>ok</name> <operator>=</operator> <literal type="boolean">false</literal></expr>;</return></block></then>
    <elseif>else <if>if <condition>(<expr><call><name><name>ps</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call> <operator>==</operator> <literal type="number">1</literal></expr>)</condition><then><block>{
        <expr_stmt><expr><call><name>uncheckedEnqueue</name><argument_list>(<argument><expr><name><name>ps</name><index>[<expr><literal type="number">0</literal></expr>]</index></name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
        <return>return <expr><name>ok</name> <operator>=</operator> <operator>(</operator><call><name>propagate</name><argument_list>()</argument_list></call> <operator>==</operator> <name>CRef_Undef</name><operator>)</operator></expr>;</return>
    }</block></then></if></elseif><else>else<block>{
        <decl_stmt><decl><type><name>CRef</name></type> <name>cr</name> <init>= <expr><call><name><name>ca</name><operator>.</operator><name>alloc</name></name><argument_list>(<argument><expr><name>ps</name></expr></argument>, <argument><expr><literal type="boolean">false</literal></expr></argument>)</argument_list></call></expr></init></decl>;</decl_stmt>
        <expr_stmt><expr><call><name><name>clauses</name><operator>.</operator><name>push</name></name><argument_list>(<argument><expr><name>cr</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
        <expr_stmt><expr><call><name>attachClause</name><argument_list>(<argument><expr><name>cr</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    }</block></else></if>

    <return>return <expr><literal type="boolean">true</literal></expr>;</return>
}</block></function>


<function><type><name>void</name></type> <name><name>Solver</name><operator>::</operator><name>attachClause</name></name><parameter_list>(<parameter><decl><type><name>CRef</name></type> <name>cr</name></decl></parameter>)</parameter_list> <block>{
    <decl_stmt><decl><type><specifier>const</specifier> <name>Clause</name><modifier>&amp;</modifier></type> <name>c</name> <init>= <expr><name><name>ca</name><index>[<expr><name>cr</name></expr>]</index></name></expr></init></decl>;</decl_stmt>
    <expr_stmt><expr><call><name>assert</name><argument_list>(<argument><expr><call><name><name>c</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call> <operator>&gt;</operator> <literal type="number">1</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <expr_stmt><expr><name><name>watches</name><index>[<expr><operator>~</operator><name><name>c</name><index>[<expr><literal type="number">0</literal></expr>]</index></name></expr>]</index></name><operator>.</operator><call><name>push</name><argument_list>(<argument><expr><call><name>Watcher</name><argument_list>(<argument><expr><name>cr</name></expr></argument>, <argument><expr><name><name>c</name><index>[<expr><literal type="number">1</literal></expr>]</index></name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <expr_stmt><expr><name><name>watches</name><index>[<expr><operator>~</operator><name><name>c</name><index>[<expr><literal type="number">1</literal></expr>]</index></name></expr>]</index></name><operator>.</operator><call><name>push</name><argument_list>(<argument><expr><call><name>Watcher</name><argument_list>(<argument><expr><name>cr</name></expr></argument>, <argument><expr><name><name>c</name><index>[<expr><literal type="number">0</literal></expr>]</index></name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <if>if <condition>(<expr><call><name><name>c</name><operator>.</operator><name>learnt</name></name><argument_list>()</argument_list></call></expr>)</condition><then> <block type="pseudo"><expr_stmt><expr><name>learnts_literals</name> <operator>+=</operator> <call><name><name>c</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</expr_stmt></block></then>
    <else>else            <block type="pseudo"><expr_stmt><expr><name>clauses_literals</name> <operator>+=</operator> <call><name><name>c</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</expr_stmt></block></else></if> }</block></function>


<function><type><name>void</name></type> <name><name>Solver</name><operator>::</operator><name>detachClause</name></name><parameter_list>(<parameter><decl><type><name>CRef</name></type> <name>cr</name></decl></parameter>, <parameter><decl><type><name>bool</name></type> <name>strict</name></decl></parameter>)</parameter_list> <block>{
    <decl_stmt><decl><type><specifier>const</specifier> <name>Clause</name><modifier>&amp;</modifier></type> <name>c</name> <init>= <expr><name><name>ca</name><index>[<expr><name>cr</name></expr>]</index></name></expr></init></decl>;</decl_stmt>
    <expr_stmt><expr><call><name>assert</name><argument_list>(<argument><expr><call><name><name>c</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call> <operator>&gt;</operator> <literal type="number">1</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    
    <if>if <condition>(<expr><name>strict</name></expr>)</condition><then><block>{
        <expr_stmt><expr><call><name>remove</name><argument_list>(<argument><expr><name><name>watches</name><index>[<expr><operator>~</operator><name><name>c</name><index>[<expr><literal type="number">0</literal></expr>]</index></name></expr>]</index></name></expr></argument>, <argument><expr><call><name>Watcher</name><argument_list>(<argument><expr><name>cr</name></expr></argument>, <argument><expr><name><name>c</name><index>[<expr><literal type="number">1</literal></expr>]</index></name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call></expr>;</expr_stmt>
        <expr_stmt><expr><call><name>remove</name><argument_list>(<argument><expr><name><name>watches</name><index>[<expr><operator>~</operator><name><name>c</name><index>[<expr><literal type="number">1</literal></expr>]</index></name></expr>]</index></name></expr></argument>, <argument><expr><call><name>Watcher</name><argument_list>(<argument><expr><name>cr</name></expr></argument>, <argument><expr><name><name>c</name><index>[<expr><literal type="number">0</literal></expr>]</index></name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    }</block></then><else>else<block>{
        <comment type="line">// Lazy detaching: (NOTE! Must clean all watcher lists before garbage collecting this clause)</comment>
        <expr_stmt><expr><call><name><name>watches</name><operator>.</operator><name>smudge</name></name><argument_list>(<argument><expr><operator>~</operator><name><name>c</name><index>[<expr><literal type="number">0</literal></expr>]</index></name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
        <expr_stmt><expr><call><name><name>watches</name><operator>.</operator><name>smudge</name></name><argument_list>(<argument><expr><operator>~</operator><name><name>c</name><index>[<expr><literal type="number">1</literal></expr>]</index></name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    }</block></else></if>

    <if>if <condition>(<expr><call><name><name>c</name><operator>.</operator><name>learnt</name></name><argument_list>()</argument_list></call></expr>)</condition><then> <block type="pseudo"><expr_stmt><expr><name>learnts_literals</name> <operator>-=</operator> <call><name><name>c</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</expr_stmt></block></then>
    <else>else            <block type="pseudo"><expr_stmt><expr><name>clauses_literals</name> <operator>-=</operator> <call><name><name>c</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</expr_stmt></block></else></if> }</block></function>


<function><type><name>void</name></type> <name><name>Solver</name><operator>::</operator><name>removeClause</name></name><parameter_list>(<parameter><decl><type><name>CRef</name></type> <name>cr</name></decl></parameter>)</parameter_list> <block>{
    <decl_stmt><decl><type><name>Clause</name><modifier>&amp;</modifier></type> <name>c</name> <init>= <expr><name><name>ca</name><index>[<expr><name>cr</name></expr>]</index></name></expr></init></decl>;</decl_stmt>
    <expr_stmt><expr><call><name>detachClause</name><argument_list>(<argument><expr><name>cr</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <comment type="line">// Don't leave pointers to free'd memory!</comment>
    <if>if <condition>(<expr><call><name>locked</name><argument_list>(<argument><expr><name>c</name></expr></argument>)</argument_list></call></expr>)</condition><then> <block type="pseudo"><expr_stmt><expr><name><name>vardata</name><index>[<expr><call><name>var</name><argument_list>(<argument><expr><name><name>c</name><index>[<expr><literal type="number">0</literal></expr>]</index></name></expr></argument>)</argument_list></call></expr>]</index></name><operator>.</operator><name>reason</name> <operator>=</operator> <name>CRef_Undef</name></expr>;</expr_stmt></block></then></if>
    <expr_stmt><expr><call><name><name>c</name><operator>.</operator><name>mark</name></name><argument_list>(<argument><expr><literal type="number">1</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt> 
    <expr_stmt><expr><call><name><name>ca</name><operator>.</operator><name>free</name></name><argument_list>(<argument><expr><name>cr</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
}</block></function>


<function><type><name>bool</name></type> <name><name>Solver</name><operator>::</operator><name>satisfied</name></name><parameter_list>(<parameter><decl><type><specifier>const</specifier> <name>Clause</name><modifier>&amp;</modifier></type> <name>c</name></decl></parameter>)</parameter_list> <specifier>const</specifier> <block>{
    <for>for <control>(<init><decl><type><name>int</name></type> <name>i</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</init> <condition><expr><name>i</name> <operator>&lt;</operator> <call><name><name>c</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>i</name><operator>++</operator></expr></incr>)</control>
        <block type="pseudo"><if>if <condition>(<expr><call><name>value</name><argument_list>(<argument><expr><name><name>c</name><index>[<expr><name>i</name></expr>]</index></name></expr></argument>)</argument_list></call> <operator>==</operator> <name>l_True</name></expr>)</condition><then>
            <block type="pseudo"><return>return <expr><literal type="boolean">true</literal></expr>;</return></block></then></if></block></for>
    <return>return <expr><literal type="boolean">false</literal></expr>;</return> }</block></function>


<comment type="line">// Revert to the state at given level (keeping all assignment at 'level' but not beyond).</comment>
<comment type="line">//</comment>
<function><type><name>void</name></type> <name><name>Solver</name><operator>::</operator><name>cancelUntil</name></name><parameter_list>(<parameter><decl><type><name>int</name></type> <name>level</name></decl></parameter>)</parameter_list> <block>{
    <if>if <condition>(<expr><call><name>decisionLevel</name><argument_list>()</argument_list></call> <operator>&gt;</operator> <name>level</name></expr>)</condition><then><block>{
        <for>for <control>(<init><decl><type><name>int</name></type> <name>c</name> <init>= <expr><call><name><name>trail</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call><operator>-</operator><literal type="number">1</literal></expr></init></decl>;</init> <condition><expr><name>c</name> <operator>&gt;=</operator> <name><name>trail_lim</name><index>[<expr><name>level</name></expr>]</index></name></expr>;</condition> <incr><expr><name>c</name><operator>--</operator></expr></incr>)</control><block>{
            <decl_stmt><decl><type><name>Var</name></type>      <name>x</name>  <init>= <expr><call><name>var</name><argument_list>(<argument><expr><name><name>trail</name><index>[<expr><name>c</name></expr>]</index></name></expr></argument>)</argument_list></call></expr></init></decl>;</decl_stmt>
            <expr_stmt><expr><name><name>assigns</name> <index>[<expr><name>x</name></expr>]</index></name> <operator>=</operator> <name>l_Undef</name></expr>;</expr_stmt>
            <if>if <condition>(<expr><name>phase_saving</name> <operator>&gt;</operator> <literal type="number">1</literal> <operator>||</operator> <operator>(</operator><name>phase_saving</name> <operator>==</operator> <literal type="number">1</literal><operator>)</operator> <operator>&amp;&amp;</operator> <name>c</name> <operator>&gt;</operator> <call><name><name>trail_lim</name><operator>.</operator><name>last</name></name><argument_list>()</argument_list></call></expr>)</condition><then>
                <block type="pseudo"><expr_stmt><expr><name><name>polarity</name><index>[<expr><name>x</name></expr>]</index></name> <operator>=</operator> <call><name>sign</name><argument_list>(<argument><expr><name><name>trail</name><index>[<expr><name>c</name></expr>]</index></name></expr></argument>)</argument_list></call></expr>;</expr_stmt></block></then></if>
            <expr_stmt><expr><call><name>insertVarOrder</name><argument_list>(<argument><expr><name>x</name></expr></argument>)</argument_list></call></expr>;</expr_stmt> }</block></for>
        <expr_stmt><expr><name>qhead</name> <operator>=</operator> <name><name>trail_lim</name><index>[<expr><name>level</name></expr>]</index></name></expr>;</expr_stmt>
        <expr_stmt><expr><call><name><name>trail</name><operator>.</operator><name>shrink</name></name><argument_list>(<argument><expr><call><name><name>trail</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call> <operator>-</operator> <name><name>trail_lim</name><index>[<expr><name>level</name></expr>]</index></name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
        <expr_stmt><expr><call><name><name>trail_lim</name><operator>.</operator><name>shrink</name></name><argument_list>(<argument><expr><call><name><name>trail_lim</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call> <operator>-</operator> <name>level</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    }</block></then></if> }</block></function>


<comment type="line">//=================================================================================================</comment>
<comment type="line">// Major methods:</comment>


<function><type><name>Lit</name></type> <name><name>Solver</name><operator>::</operator><name>pickBranchLit</name></name><parameter_list>()</parameter_list>
<block>{
    <decl_stmt><decl><type><name>Var</name></type> <name>next</name> <init>= <expr><name>var_Undef</name></expr></init></decl>;</decl_stmt>

    <comment type="line">// Random decision:</comment>
    <if>if <condition>(<expr><call><name>drand</name><argument_list>(<argument><expr><name>random_seed</name></expr></argument>)</argument_list></call> <operator>&lt;</operator> <name>random_var_freq</name> <operator>&amp;&amp;</operator> <operator>!</operator><call><name><name>order_heap</name><operator>.</operator><name>empty</name></name><argument_list>()</argument_list></call></expr>)</condition><then><block>{
        <expr_stmt><expr><name>next</name> <operator>=</operator> <name><name>order_heap</name><index>[<expr><call><name>irand</name><argument_list>(<argument><expr><name>random_seed</name></expr></argument>,<argument><expr><call><name><name>order_heap</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr></argument>)</argument_list></call></expr>]</index></name></expr>;</expr_stmt>
        <if>if <condition>(<expr><call><name>value</name><argument_list>(<argument><expr><name>next</name></expr></argument>)</argument_list></call> <operator>==</operator> <name>l_Undef</name> <operator>&amp;&amp;</operator> <name><name>decision</name><index>[<expr><name>next</name></expr>]</index></name></expr>)</condition><then>
            <block type="pseudo"><expr_stmt><expr><name>rnd_decisions</name><operator>++</operator></expr>;</expr_stmt></block></then></if> }</block></then></if>

    <comment type="line">// Activity based decision:</comment>
    <while>while <condition>(<expr><name>next</name> <operator>==</operator> <name>var_Undef</name> <operator>||</operator> <call><name>value</name><argument_list>(<argument><expr><name>next</name></expr></argument>)</argument_list></call> <operator>!=</operator> <name>l_Undef</name> <operator>||</operator> <operator>!</operator><name><name>decision</name><index>[<expr><name>next</name></expr>]</index></name></expr>)</condition>
        <block type="pseudo"><if>if <condition>(<expr><call><name><name>order_heap</name><operator>.</operator><name>empty</name></name><argument_list>()</argument_list></call></expr>)</condition><then><block>{
            <expr_stmt><expr><name>next</name> <operator>=</operator> <name>var_Undef</name></expr>;</expr_stmt>
            <break>break;</break>
        }</block></then><else>else
            <block type="pseudo"><expr_stmt><expr><name>next</name> <operator>=</operator> <call><name><name>order_heap</name><operator>.</operator><name>removeMin</name></name><argument_list>()</argument_list></call></expr>;</expr_stmt></block></else></if></block></while>

    <return>return <expr><ternary><condition><expr><name>next</name> <operator>==</operator> <name>var_Undef</name></expr> ?</condition><then> <expr><name>lit_Undef</name></expr> </then><else>: <expr><call><name>mkLit</name><argument_list>(<argument><expr><name>next</name></expr></argument>, <argument><expr><ternary><condition><expr><name>rnd_pol</name></expr> ?</condition><then> <expr><call><name>drand</name><argument_list>(<argument><expr><name>random_seed</name></expr></argument>)</argument_list></call> <operator>&lt;</operator> <literal type="number">0.5</literal></expr> </then><else>: <expr><name><name>polarity</name><index>[<expr><name>next</name></expr>]</index></name></expr></else></ternary></expr></argument>)</argument_list></call></expr></else></ternary></expr>;</return>
}</block></function>


<comment type="block">/*_________________________________________________________________________________________________
|
|  analyze : (confl : Clause*) (out_learnt : vec&lt;Lit&gt;&amp;) (out_btlevel : int&amp;)  -&gt;  [void]
|  
|  Description:
|    Analyze conflict and produce a reason clause.
|  
|    Pre-conditions:
|      * 'out_learnt' is assumed to be cleared.
|      * Current decision level must be greater than root level.
|  
|    Post-conditions:
|      * 'out_learnt[0]' is the asserting literal at level 'out_btlevel'.
|      * If out_learnt.size() &gt; 1 then 'out_learnt[1]' has the greatest decision level of the 
|        rest of literals. There may be others from the same level though.
|  
|________________________________________________________________________________________________@*/</comment>
<function><type><name>void</name></type> <name><name>Solver</name><operator>::</operator><name>analyze</name></name><parameter_list>(<parameter><decl><type><name>CRef</name></type> <name>confl</name></decl></parameter>, <parameter><decl><type><name><name>vec</name><argument_list type="generic">&lt;<argument><expr><name>Lit</name></expr></argument>&gt;</argument_list></name><modifier>&amp;</modifier></type> <name>out_learnt</name></decl></parameter>, <parameter><decl><type><name>int</name><modifier>&amp;</modifier></type> <name>out_btlevel</name></decl></parameter>)</parameter_list>
<block>{
    <decl_stmt><decl><type><name>int</name></type> <name>pathC</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</decl_stmt>
    <decl_stmt><decl><type><name>Lit</name></type> <name>p</name>     <init>= <expr><name>lit_Undef</name></expr></init></decl>;</decl_stmt>

    <comment type="line">// Generate conflict clause:</comment>
    <comment type="line">//</comment>
    <expr_stmt><expr><call><name><name>out_learnt</name><operator>.</operator><name>push</name></name><argument_list>()</argument_list></call></expr>;</expr_stmt>      <comment type="line">// (leave room for the asserting literal)</comment>
    <decl_stmt><decl><type><name>int</name></type> <name>index</name>   <init>= <expr><call><name><name>trail</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call> <operator>-</operator> <literal type="number">1</literal></expr></init></decl>;</decl_stmt>

    <do>do<block>{
        <expr_stmt><expr><call><name>assert</name><argument_list>(<argument><expr><name>confl</name> <operator>!=</operator> <name>CRef_Undef</name></expr></argument>)</argument_list></call></expr>;</expr_stmt> <comment type="line">// (otherwise should be UIP)</comment>
        <decl_stmt><decl><type><name>Clause</name><modifier>&amp;</modifier></type> <name>c</name> <init>= <expr><name><name>ca</name><index>[<expr><name>confl</name></expr>]</index></name></expr></init></decl>;</decl_stmt>

        <if>if <condition>(<expr><call><name><name>c</name><operator>.</operator><name>learnt</name></name><argument_list>()</argument_list></call></expr>)</condition><then>
            <block type="pseudo"><expr_stmt><expr><call><name>claBumpActivity</name><argument_list>(<argument><expr><name>c</name></expr></argument>)</argument_list></call></expr>;</expr_stmt></block></then></if>

        <for>for <control>(<init><decl><type><name>int</name></type> <name>j</name> <init>= <expr><ternary><condition><expr><operator>(</operator><name>p</name> <operator>==</operator> <name>lit_Undef</name><operator>)</operator></expr> ?</condition><then> <expr><literal type="number">0</literal></expr> </then><else>: <expr><literal type="number">1</literal></expr></else></ternary></expr></init></decl>;</init> <condition><expr><name>j</name> <operator>&lt;</operator> <call><name><name>c</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>j</name><operator>++</operator></expr></incr>)</control><block>{
            <decl_stmt><decl><type><name>Lit</name></type> <name>q</name> <init>= <expr><name><name>c</name><index>[<expr><name>j</name></expr>]</index></name></expr></init></decl>;</decl_stmt>

            <if>if <condition>(<expr><operator>!</operator><name><name>seen</name><index>[<expr><call><name>var</name><argument_list>(<argument><expr><name>q</name></expr></argument>)</argument_list></call></expr>]</index></name> <operator>&amp;&amp;</operator> <call><name>level</name><argument_list>(<argument><expr><call><name>var</name><argument_list>(<argument><expr><name>q</name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call> <operator>&gt;</operator> <literal type="number">0</literal></expr>)</condition><then><block>{
                <expr_stmt><expr><call><name>varBumpActivity</name><argument_list>(<argument><expr><call><name>var</name><argument_list>(<argument><expr><name>q</name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call></expr>;</expr_stmt>
                <expr_stmt><expr><name><name>seen</name><index>[<expr><call><name>var</name><argument_list>(<argument><expr><name>q</name></expr></argument>)</argument_list></call></expr>]</index></name> <operator>=</operator> <literal type="number">1</literal></expr>;</expr_stmt>
                <if>if <condition>(<expr><call><name>level</name><argument_list>(<argument><expr><call><name>var</name><argument_list>(<argument><expr><name>q</name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call> <operator>&gt;=</operator> <call><name>decisionLevel</name><argument_list>()</argument_list></call></expr>)</condition><then>
                    <block type="pseudo"><expr_stmt><expr><name>pathC</name><operator>++</operator></expr>;</expr_stmt></block></then>
                <else>else
                    <block type="pseudo"><expr_stmt><expr><call><name><name>out_learnt</name><operator>.</operator><name>push</name></name><argument_list>(<argument><expr><name>q</name></expr></argument>)</argument_list></call></expr>;</expr_stmt></block></else></if>
            }</block></then></if>
        }</block></for>
        
        <comment type="line">// Select next clause to look at:</comment>
        <while>while <condition>(<expr><operator>!</operator><name><name>seen</name><index>[<expr><call><name>var</name><argument_list>(<argument><expr><name><name>trail</name><index>[<expr><name>index</name><operator>--</operator></expr>]</index></name></expr></argument>)</argument_list></call></expr>]</index></name></expr>)</condition><block type="pseudo"><empty_stmt>;</empty_stmt></block></while>
        <expr_stmt><expr><name>p</name>     <operator>=</operator> <name><name>trail</name><index>[<expr><name>index</name><operator>+</operator><literal type="number">1</literal></expr>]</index></name></expr>;</expr_stmt>
        <expr_stmt><expr><name>confl</name> <operator>=</operator> <call><name>reason</name><argument_list>(<argument><expr><call><name>var</name><argument_list>(<argument><expr><name>p</name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call></expr>;</expr_stmt>
        <expr_stmt><expr><name><name>seen</name><index>[<expr><call><name>var</name><argument_list>(<argument><expr><name>p</name></expr></argument>)</argument_list></call></expr>]</index></name> <operator>=</operator> <literal type="number">0</literal></expr>;</expr_stmt>
        <expr_stmt><expr><name>pathC</name><operator>--</operator></expr>;</expr_stmt>

    }</block>while <condition>(<expr><name>pathC</name> <operator>&gt;</operator> <literal type="number">0</literal></expr>)</condition>;</do>
    <expr_stmt><expr><name><name>out_learnt</name><index>[<expr><literal type="number">0</literal></expr>]</index></name> <operator>=</operator> <operator>~</operator><name>p</name></expr>;</expr_stmt>

    <comment type="line">// Simplify conflict clause:</comment>
    <comment type="line">//</comment>
    <decl_stmt><decl><type><name>int</name></type> <name>i</name></decl>, <decl><type ref="prev"/><name>j</name></decl>;</decl_stmt>
    <expr_stmt><expr><call><name><name>out_learnt</name><operator>.</operator><name>copyTo</name></name><argument_list>(<argument><expr><name>analyze_toclear</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <if>if <condition>(<expr><name>ccmin_mode</name> <operator>==</operator> <literal type="number">2</literal></expr>)</condition><then><block>{
        <decl_stmt><decl><type><name>uint32_t</name></type> <name>abstract_level</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</decl_stmt>
        <for>for <control>(<init><expr><name>i</name> <operator>=</operator> <literal type="number">1</literal></expr>;</init> <condition><expr><name>i</name> <operator>&lt;</operator> <call><name><name>out_learnt</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>i</name><operator>++</operator></expr></incr>)</control>
            <block type="pseudo"><expr_stmt><expr><name>abstract_level</name> <operator>|=</operator> <call><name>abstractLevel</name><argument_list>(<argument><expr><call><name>var</name><argument_list>(<argument><expr><name><name>out_learnt</name><index>[<expr><name>i</name></expr>]</index></name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call></expr>;</expr_stmt></block></for> <comment type="line">// (maintain an abstraction of levels involved in conflict)</comment>

        <for>for <control>(<init><expr><name>i</name> <operator>=</operator> <name>j</name> <operator>=</operator> <literal type="number">1</literal></expr>;</init> <condition><expr><name>i</name> <operator>&lt;</operator> <call><name><name>out_learnt</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>i</name><operator>++</operator></expr></incr>)</control>
            <block type="pseudo"><if>if <condition>(<expr><call><name>reason</name><argument_list>(<argument><expr><call><name>var</name><argument_list>(<argument><expr><name><name>out_learnt</name><index>[<expr><name>i</name></expr>]</index></name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call> <operator>==</operator> <name>CRef_Undef</name> <operator>||</operator> <operator>!</operator><call><name>litRedundant</name><argument_list>(<argument><expr><name><name>out_learnt</name><index>[<expr><name>i</name></expr>]</index></name></expr></argument>, <argument><expr><name>abstract_level</name></expr></argument>)</argument_list></call></expr>)</condition><then>
                <block type="pseudo"><expr_stmt><expr><name><name>out_learnt</name><index>[<expr><name>j</name><operator>++</operator></expr>]</index></name> <operator>=</operator> <name><name>out_learnt</name><index>[<expr><name>i</name></expr>]</index></name></expr>;</expr_stmt></block></then></if></block></for>
        
    }</block></then><elseif>else <if>if <condition>(<expr><name>ccmin_mode</name> <operator>==</operator> <literal type="number">1</literal></expr>)</condition><then><block>{
        <for>for <control>(<init><expr><name>i</name> <operator>=</operator> <name>j</name> <operator>=</operator> <literal type="number">1</literal></expr>;</init> <condition><expr><name>i</name> <operator>&lt;</operator> <call><name><name>out_learnt</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>i</name><operator>++</operator></expr></incr>)</control><block>{
            <decl_stmt><decl><type><name>Var</name></type> <name>x</name> <init>= <expr><call><name>var</name><argument_list>(<argument><expr><name><name>out_learnt</name><index>[<expr><name>i</name></expr>]</index></name></expr></argument>)</argument_list></call></expr></init></decl>;</decl_stmt>

            <if>if <condition>(<expr><call><name>reason</name><argument_list>(<argument><expr><name>x</name></expr></argument>)</argument_list></call> <operator>==</operator> <name>CRef_Undef</name></expr>)</condition><then>
                <block type="pseudo"><expr_stmt><expr><name><name>out_learnt</name><index>[<expr><name>j</name><operator>++</operator></expr>]</index></name> <operator>=</operator> <name><name>out_learnt</name><index>[<expr><name>i</name></expr>]</index></name></expr>;</expr_stmt></block></then>
            <else>else<block>{
                <decl_stmt><decl><type><name>Clause</name><modifier>&amp;</modifier></type> <name>c</name> <init>= <expr><name><name>ca</name><index>[<expr><call><name>reason</name><argument_list>(<argument><expr><call><name>var</name><argument_list>(<argument><expr><name><name>out_learnt</name><index>[<expr><name>i</name></expr>]</index></name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call></expr>]</index></name></expr></init></decl>;</decl_stmt>
                <for>for <control>(<init><decl><type><name>int</name></type> <name>k</name> <init>= <expr><literal type="number">1</literal></expr></init></decl>;</init> <condition><expr><name>k</name> <operator>&lt;</operator> <call><name><name>c</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>k</name><operator>++</operator></expr></incr>)</control>
                    <block type="pseudo"><if>if <condition>(<expr><operator>!</operator><name><name>seen</name><index>[<expr><call><name>var</name><argument_list>(<argument><expr><name><name>c</name><index>[<expr><name>k</name></expr>]</index></name></expr></argument>)</argument_list></call></expr>]</index></name> <operator>&amp;&amp;</operator> <call><name>level</name><argument_list>(<argument><expr><call><name>var</name><argument_list>(<argument><expr><name><name>c</name><index>[<expr><name>k</name></expr>]</index></name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call> <operator>&gt;</operator> <literal type="number">0</literal></expr>)</condition><then><block>{
                        <expr_stmt><expr><name><name>out_learnt</name><index>[<expr><name>j</name><operator>++</operator></expr>]</index></name> <operator>=</operator> <name><name>out_learnt</name><index>[<expr><name>i</name></expr>]</index></name></expr>;</expr_stmt>
                        <break>break;</break> }</block></then></if></block></for>
            }</block></else></if>
        }</block></for>
    }</block></then></if></elseif><else>else
        <block type="pseudo"><expr_stmt><expr><name>i</name> <operator>=</operator> <name>j</name> <operator>=</operator> <call><name><name>out_learnt</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</expr_stmt></block></else></if>

    <expr_stmt><expr><name>max_literals</name> <operator>+=</operator> <call><name><name>out_learnt</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</expr_stmt>
    <expr_stmt><expr><call><name><name>out_learnt</name><operator>.</operator><name>shrink</name></name><argument_list>(<argument><expr><name>i</name> <operator>-</operator> <name>j</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <expr_stmt><expr><name>tot_literals</name> <operator>+=</operator> <call><name><name>out_learnt</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</expr_stmt>

    <comment type="line">// Find correct backtrack level:</comment>
    <comment type="line">//</comment>
    <if>if <condition>(<expr><call><name><name>out_learnt</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call> <operator>==</operator> <literal type="number">1</literal></expr>)</condition><then>
        <block type="pseudo"><expr_stmt><expr><name>out_btlevel</name> <operator>=</operator> <literal type="number">0</literal></expr>;</expr_stmt></block></then>
    <else>else<block>{
        <decl_stmt><decl><type><name>int</name></type> <name>max_i</name> <init>= <expr><literal type="number">1</literal></expr></init></decl>;</decl_stmt>
        <comment type="line">// Find the first literal assigned at the next-highest level:</comment>
        <for>for <control>(<init><decl><type><name>int</name></type> <name>i</name> <init>= <expr><literal type="number">2</literal></expr></init></decl>;</init> <condition><expr><name>i</name> <operator>&lt;</operator> <call><name><name>out_learnt</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>i</name><operator>++</operator></expr></incr>)</control>
            <block type="pseudo"><if>if <condition>(<expr><call><name>level</name><argument_list>(<argument><expr><call><name>var</name><argument_list>(<argument><expr><name><name>out_learnt</name><index>[<expr><name>i</name></expr>]</index></name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call> <operator>&gt;</operator> <call><name>level</name><argument_list>(<argument><expr><call><name>var</name><argument_list>(<argument><expr><name><name>out_learnt</name><index>[<expr><name>max_i</name></expr>]</index></name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call></expr>)</condition><then>
                <block type="pseudo"><expr_stmt><expr><name>max_i</name> <operator>=</operator> <name>i</name></expr>;</expr_stmt></block></then></if></block></for>
        <comment type="line">// Swap-in this literal at index 1:</comment>
        <decl_stmt><decl><type><name>Lit</name></type> <name>p</name>             <init>= <expr><name><name>out_learnt</name><index>[<expr><name>max_i</name></expr>]</index></name></expr></init></decl>;</decl_stmt>
        <expr_stmt><expr><name><name>out_learnt</name><index>[<expr><name>max_i</name></expr>]</index></name> <operator>=</operator> <name><name>out_learnt</name><index>[<expr><literal type="number">1</literal></expr>]</index></name></expr>;</expr_stmt>
        <expr_stmt><expr><name><name>out_learnt</name><index>[<expr><literal type="number">1</literal></expr>]</index></name>     <operator>=</operator> <name>p</name></expr>;</expr_stmt>
        <expr_stmt><expr><name>out_btlevel</name>       <operator>=</operator> <call><name>level</name><argument_list>(<argument><expr><call><name>var</name><argument_list>(<argument><expr><name>p</name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    }</block></else></if>

    <for>for <control>(<init><decl><type><name>int</name></type> <name>j</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</init> <condition><expr><name>j</name> <operator>&lt;</operator> <call><name><name>analyze_toclear</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>j</name><operator>++</operator></expr></incr>)</control> <block type="pseudo"><expr_stmt><expr><name><name>seen</name><index>[<expr><call><name>var</name><argument_list>(<argument><expr><name><name>analyze_toclear</name><index>[<expr><name>j</name></expr>]</index></name></expr></argument>)</argument_list></call></expr>]</index></name> <operator>=</operator> <literal type="number">0</literal></expr>;</expr_stmt></block></for>    <comment type="line">// ('seen[]' is now cleared)</comment>
}</block></function>


<comment type="line">// Check if 'p' can be removed. 'abstract_levels' is used to abort early if the algorithm is</comment>
<comment type="line">// visiting literals at levels that cannot be removed later.</comment>
<function><type><name>bool</name></type> <name><name>Solver</name><operator>::</operator><name>litRedundant</name></name><parameter_list>(<parameter><decl><type><name>Lit</name></type> <name>p</name></decl></parameter>, <parameter><decl><type><name>uint32_t</name></type> <name>abstract_levels</name></decl></parameter>)</parameter_list>
<block>{
    <expr_stmt><expr><call><name><name>analyze_stack</name><operator>.</operator><name>clear</name></name><argument_list>()</argument_list></call></expr>;</expr_stmt> <expr_stmt><expr><call><name><name>analyze_stack</name><operator>.</operator><name>push</name></name><argument_list>(<argument><expr><name>p</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <decl_stmt><decl><type><name>int</name></type> <name>top</name> <init>= <expr><call><name><name>analyze_toclear</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr></init></decl>;</decl_stmt>
    <while>while <condition>(<expr><call><name><name>analyze_stack</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call> <operator>&gt;</operator> <literal type="number">0</literal></expr>)</condition><block>{
        <expr_stmt><expr><call><name>assert</name><argument_list>(<argument><expr><call><name>reason</name><argument_list>(<argument><expr><call><name>var</name><argument_list>(<argument><expr><call><name><name>analyze_stack</name><operator>.</operator><name>last</name></name><argument_list>()</argument_list></call></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call> <operator>!=</operator> <name>CRef_Undef</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
        <decl_stmt><decl><type><name>Clause</name><modifier>&amp;</modifier></type> <name>c</name> <init>= <expr><name><name>ca</name><index>[<expr><call><name>reason</name><argument_list>(<argument><expr><call><name>var</name><argument_list>(<argument><expr><call><name><name>analyze_stack</name><operator>.</operator><name>last</name></name><argument_list>()</argument_list></call></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call></expr>]</index></name></expr></init></decl>;</decl_stmt> <expr_stmt><expr><call><name><name>analyze_stack</name><operator>.</operator><name>pop</name></name><argument_list>()</argument_list></call></expr>;</expr_stmt>

        <for>for <control>(<init><decl><type><name>int</name></type> <name>i</name> <init>= <expr><literal type="number">1</literal></expr></init></decl>;</init> <condition><expr><name>i</name> <operator>&lt;</operator> <call><name><name>c</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>i</name><operator>++</operator></expr></incr>)</control><block>{
            <decl_stmt><decl><type><name>Lit</name></type> <name>p</name>  <init>= <expr><name><name>c</name><index>[<expr><name>i</name></expr>]</index></name></expr></init></decl>;</decl_stmt>
            <if>if <condition>(<expr><operator>!</operator><name><name>seen</name><index>[<expr><call><name>var</name><argument_list>(<argument><expr><name>p</name></expr></argument>)</argument_list></call></expr>]</index></name> <operator>&amp;&amp;</operator> <call><name>level</name><argument_list>(<argument><expr><call><name>var</name><argument_list>(<argument><expr><name>p</name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call> <operator>&gt;</operator> <literal type="number">0</literal></expr>)</condition><then><block>{
                <if>if <condition>(<expr><call><name>reason</name><argument_list>(<argument><expr><call><name>var</name><argument_list>(<argument><expr><name>p</name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call> <operator>!=</operator> <name>CRef_Undef</name> <operator>&amp;&amp;</operator> <operator>(</operator><call><name>abstractLevel</name><argument_list>(<argument><expr><call><name>var</name><argument_list>(<argument><expr><name>p</name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call> <operator>&amp;</operator> <name>abstract_levels</name><operator>)</operator> <operator>!=</operator> <literal type="number">0</literal></expr>)</condition><then><block>{
                    <expr_stmt><expr><name><name>seen</name><index>[<expr><call><name>var</name><argument_list>(<argument><expr><name>p</name></expr></argument>)</argument_list></call></expr>]</index></name> <operator>=</operator> <literal type="number">1</literal></expr>;</expr_stmt>
                    <expr_stmt><expr><call><name><name>analyze_stack</name><operator>.</operator><name>push</name></name><argument_list>(<argument><expr><name>p</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
                    <expr_stmt><expr><call><name><name>analyze_toclear</name><operator>.</operator><name>push</name></name><argument_list>(<argument><expr><name>p</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
                }</block></then><else>else<block>{
                    <for>for <control>(<init><decl><type><name>int</name></type> <name>j</name> <init>= <expr><name>top</name></expr></init></decl>;</init> <condition><expr><name>j</name> <operator>&lt;</operator> <call><name><name>analyze_toclear</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>j</name><operator>++</operator></expr></incr>)</control>
                        <block type="pseudo"><expr_stmt><expr><name><name>seen</name><index>[<expr><call><name>var</name><argument_list>(<argument><expr><name><name>analyze_toclear</name><index>[<expr><name>j</name></expr>]</index></name></expr></argument>)</argument_list></call></expr>]</index></name> <operator>=</operator> <literal type="number">0</literal></expr>;</expr_stmt></block></for>
                    <expr_stmt><expr><call><name><name>analyze_toclear</name><operator>.</operator><name>shrink</name></name><argument_list>(<argument><expr><call><name><name>analyze_toclear</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call> <operator>-</operator> <name>top</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
                    <return>return <expr><literal type="boolean">false</literal></expr>;</return>
                }</block></else></if>
            }</block></then></if>
        }</block></for>
    }</block></while>

    <return>return <expr><literal type="boolean">true</literal></expr>;</return>
}</block></function>


<comment type="block">/*_________________________________________________________________________________________________
|
|  analyzeFinal : (p : Lit)  -&gt;  [void]
|  
|  Description:
|    Specialized analysis procedure to express the final conflict in terms of assumptions.
|    Calculates the (possibly empty) set of assumptions that led to the assignment of 'p', and
|    stores the result in 'out_conflict'.
|________________________________________________________________________________________________@*/</comment>
<function><type><name>void</name></type> <name><name>Solver</name><operator>::</operator><name>analyzeFinal</name></name><parameter_list>(<parameter><decl><type><name>Lit</name></type> <name>p</name></decl></parameter>, <parameter><decl><type><name><name>vec</name><argument_list type="generic">&lt;<argument><expr><name>Lit</name></expr></argument>&gt;</argument_list></name><modifier>&amp;</modifier></type> <name>out_conflict</name></decl></parameter>)</parameter_list>
<block>{
    <expr_stmt><expr><call><name><name>out_conflict</name><operator>.</operator><name>clear</name></name><argument_list>()</argument_list></call></expr>;</expr_stmt>
    <expr_stmt><expr><call><name><name>out_conflict</name><operator>.</operator><name>push</name></name><argument_list>(<argument><expr><name>p</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>

    <if>if <condition>(<expr><call><name>decisionLevel</name><argument_list>()</argument_list></call> <operator>==</operator> <literal type="number">0</literal></expr>)</condition><then>
        <block type="pseudo"><return>return;</return></block></then></if>

    <expr_stmt><expr><name><name>seen</name><index>[<expr><call><name>var</name><argument_list>(<argument><expr><name>p</name></expr></argument>)</argument_list></call></expr>]</index></name> <operator>=</operator> <literal type="number">1</literal></expr>;</expr_stmt>

    <for>for <control>(<init><decl><type><name>int</name></type> <name>i</name> <init>= <expr><call><name><name>trail</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call><operator>-</operator><literal type="number">1</literal></expr></init></decl>;</init> <condition><expr><name>i</name> <operator>&gt;=</operator> <name><name>trail_lim</name><index>[<expr><literal type="number">0</literal></expr>]</index></name></expr>;</condition> <incr><expr><name>i</name><operator>--</operator></expr></incr>)</control><block>{
        <decl_stmt><decl><type><name>Var</name></type> <name>x</name> <init>= <expr><call><name>var</name><argument_list>(<argument><expr><name><name>trail</name><index>[<expr><name>i</name></expr>]</index></name></expr></argument>)</argument_list></call></expr></init></decl>;</decl_stmt>
        <if>if <condition>(<expr><name><name>seen</name><index>[<expr><name>x</name></expr>]</index></name></expr>)</condition><then><block>{
            <if>if <condition>(<expr><call><name>reason</name><argument_list>(<argument><expr><name>x</name></expr></argument>)</argument_list></call> <operator>==</operator> <name>CRef_Undef</name></expr>)</condition><then><block>{
                <expr_stmt><expr><call><name>assert</name><argument_list>(<argument><expr><call><name>level</name><argument_list>(<argument><expr><name>x</name></expr></argument>)</argument_list></call> <operator>&gt;</operator> <literal type="number">0</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt>
                <expr_stmt><expr><call><name><name>out_conflict</name><operator>.</operator><name>push</name></name><argument_list>(<argument><expr><operator>~</operator><name><name>trail</name><index>[<expr><name>i</name></expr>]</index></name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
            }</block></then><else>else<block>{
                <decl_stmt><decl><type><name>Clause</name><modifier>&amp;</modifier></type> <name>c</name> <init>= <expr><name><name>ca</name><index>[<expr><call><name>reason</name><argument_list>(<argument><expr><name>x</name></expr></argument>)</argument_list></call></expr>]</index></name></expr></init></decl>;</decl_stmt>
                <for>for <control>(<init><decl><type><name>int</name></type> <name>j</name> <init>= <expr><literal type="number">1</literal></expr></init></decl>;</init> <condition><expr><name>j</name> <operator>&lt;</operator> <call><name><name>c</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>j</name><operator>++</operator></expr></incr>)</control>
                    <block type="pseudo"><if>if <condition>(<expr><call><name>level</name><argument_list>(<argument><expr><call><name>var</name><argument_list>(<argument><expr><name><name>c</name><index>[<expr><name>j</name></expr>]</index></name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call> <operator>&gt;</operator> <literal type="number">0</literal></expr>)</condition><then>
                        <block type="pseudo"><expr_stmt><expr><name><name>seen</name><index>[<expr><call><name>var</name><argument_list>(<argument><expr><name><name>c</name><index>[<expr><name>j</name></expr>]</index></name></expr></argument>)</argument_list></call></expr>]</index></name> <operator>=</operator> <literal type="number">1</literal></expr>;</expr_stmt></block></then></if></block></for>
            }</block></else></if>
            <expr_stmt><expr><name><name>seen</name><index>[<expr><name>x</name></expr>]</index></name> <operator>=</operator> <literal type="number">0</literal></expr>;</expr_stmt>
        }</block></then></if>
    }</block></for>

    <expr_stmt><expr><name><name>seen</name><index>[<expr><call><name>var</name><argument_list>(<argument><expr><name>p</name></expr></argument>)</argument_list></call></expr>]</index></name> <operator>=</operator> <literal type="number">0</literal></expr>;</expr_stmt>
}</block></function>


<function><type><name>void</name></type> <name><name>Solver</name><operator>::</operator><name>uncheckedEnqueue</name></name><parameter_list>(<parameter><decl><type><name>Lit</name></type> <name>p</name></decl></parameter>, <parameter><decl><type><name>CRef</name></type> <name>from</name></decl></parameter>)</parameter_list>
<block>{
    <expr_stmt><expr><call><name>assert</name><argument_list>(<argument><expr><call><name>value</name><argument_list>(<argument><expr><name>p</name></expr></argument>)</argument_list></call> <operator>==</operator> <name>l_Undef</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <expr_stmt><expr><name><name>assigns</name><index>[<expr><call><name>var</name><argument_list>(<argument><expr><name>p</name></expr></argument>)</argument_list></call></expr>]</index></name> <operator>=</operator> <call><name>lbool</name><argument_list>(<argument><expr><operator>!</operator><call><name>sign</name><argument_list>(<argument><expr><name>p</name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <expr_stmt><expr><name><name>vardata</name><index>[<expr><call><name>var</name><argument_list>(<argument><expr><name>p</name></expr></argument>)</argument_list></call></expr>]</index></name> <operator>=</operator> <call><name>mkVarData</name><argument_list>(<argument><expr><name>from</name></expr></argument>, <argument><expr><call><name>decisionLevel</name><argument_list>()</argument_list></call></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <expr_stmt><expr><call><name><name>trail</name><operator>.</operator><name>push_</name></name><argument_list>(<argument><expr><name>p</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
}</block></function>


<comment type="block">/*_________________________________________________________________________________________________
|
|  propagate : [void]  -&gt;  [Clause*]
|  
|  Description:
|    Propagates all enqueued facts. If a conflict arises, the conflicting clause is returned,
|    otherwise CRef_Undef.
|  
|    Post-conditions:
|      * the propagation queue is empty, even if there was a conflict.
|________________________________________________________________________________________________@*/</comment>
<function><type><name>CRef</name></type> <name><name>Solver</name><operator>::</operator><name>propagate</name></name><parameter_list>()</parameter_list>
<block>{
    <decl_stmt><decl><type><name>CRef</name></type>    <name>confl</name>     <init>= <expr><name>CRef_Undef</name></expr></init></decl>;</decl_stmt>
    <decl_stmt><decl><type><name>int</name></type>     <name>num_props</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</decl_stmt>
    <expr_stmt><expr><call><name><name>watches</name><operator>.</operator><name>cleanAll</name></name><argument_list>()</argument_list></call></expr>;</expr_stmt>

    <while>while <condition>(<expr><name>qhead</name> <operator>&lt;</operator> <call><name><name>trail</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>)</condition><block>{
        <decl_stmt><decl><type><name>Lit</name></type>            <name>p</name>   <init>= <expr><name><name>trail</name><index>[<expr><name>qhead</name><operator>++</operator></expr>]</index></name></expr></init></decl>;</decl_stmt>     <comment type="line">// 'p' is enqueued fact to propagate.</comment>
        <decl_stmt><decl><type><name><name>vec</name><argument_list type="generic">&lt;<argument><expr><name>Watcher</name></expr></argument>&gt;</argument_list></name><modifier>&amp;</modifier></type>  <name>ws</name>  <init>= <expr><name><name>watches</name><index>[<expr><name>p</name></expr>]</index></name></expr></init></decl>;</decl_stmt>
        <decl_stmt><decl><type><name>Watcher</name>        <modifier>*</modifier></type><name>i</name></decl>, <modifier>*</modifier><decl><type ref="prev"/><name>j</name></decl>, <modifier>*</modifier><decl><type ref="prev"/><name>end</name></decl>;</decl_stmt>
        <expr_stmt><expr><name>num_props</name><operator>++</operator></expr>;</expr_stmt>

        <for>for <control>(<init><expr><name>i</name> <operator>=</operator> <name>j</name> <operator>=</operator> <operator>(</operator><name>Watcher</name><operator>*</operator><operator>)</operator><name>ws</name></expr><operator>,</operator> <expr><name>end</name> <operator>=</operator> <name>i</name> <operator>+</operator> <call><name><name>ws</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</init>  <condition><expr><name>i</name> <operator>!=</operator> <name>end</name></expr>;</condition><incr/>)</control><block>{
            <comment type="line">// Try to avoid inspecting the clause:</comment>
            <decl_stmt><decl><type><name>Lit</name></type> <name>blocker</name> <init>= <expr><name><name>i</name><operator>-&gt;</operator><name>blocker</name></name></expr></init></decl>;</decl_stmt>
            <if>if <condition>(<expr><call><name>value</name><argument_list>(<argument><expr><name>blocker</name></expr></argument>)</argument_list></call> <operator>==</operator> <name>l_True</name></expr>)</condition><then><block>{
                <expr_stmt><expr><operator>*</operator><name>j</name><operator>++</operator> <operator>=</operator> <operator>*</operator><name>i</name><operator>++</operator></expr>;</expr_stmt> <continue>continue;</continue> }</block></then></if>

            <comment type="line">// Make sure the false literal is data[1]:</comment>
            <decl_stmt><decl><type><name>CRef</name></type>     <name>cr</name>        <init>= <expr><name><name>i</name><operator>-&gt;</operator><name>cref</name></name></expr></init></decl>;</decl_stmt>
            <decl_stmt><decl><type><name>Clause</name><modifier>&amp;</modifier></type>  <name>c</name>         <init>= <expr><name><name>ca</name><index>[<expr><name>cr</name></expr>]</index></name></expr></init></decl>;</decl_stmt>
            <decl_stmt><decl><type><name>Lit</name></type>      <name>false_lit</name> <init>= <expr><operator>~</operator><name>p</name></expr></init></decl>;</decl_stmt>
            <if>if <condition>(<expr><name><name>c</name><index>[<expr><literal type="number">0</literal></expr>]</index></name> <operator>==</operator> <name>false_lit</name></expr>)</condition><then>
                <block type="pseudo"><expr_stmt><expr><name><name>c</name><index>[<expr><literal type="number">0</literal></expr>]</index></name> <operator>=</operator> <name><name>c</name><index>[<expr><literal type="number">1</literal></expr>]</index></name></expr><operator>,</operator> <expr><name><name>c</name><index>[<expr><literal type="number">1</literal></expr>]</index></name> <operator>=</operator> <name>false_lit</name></expr>;</expr_stmt></block></then></if>
            <expr_stmt><expr><call><name>assert</name><argument_list>(<argument><expr><name><name>c</name><index>[<expr><literal type="number">1</literal></expr>]</index></name> <operator>==</operator> <name>false_lit</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
            <expr_stmt><expr><name>i</name><operator>++</operator></expr>;</expr_stmt>

            <comment type="line">// If 0th watch is true, then clause is already satisfied.</comment>
            <decl_stmt><decl><type><name>Lit</name></type>     <name>first</name> <init>= <expr><name><name>c</name><index>[<expr><literal type="number">0</literal></expr>]</index></name></expr></init></decl>;</decl_stmt>
            <decl_stmt><decl><type><name>Watcher</name></type> <name>w</name>     <init>= <expr><call><name>Watcher</name><argument_list>(<argument><expr><name>cr</name></expr></argument>, <argument><expr><name>first</name></expr></argument>)</argument_list></call></expr></init></decl>;</decl_stmt>
            <if>if <condition>(<expr><name>first</name> <operator>!=</operator> <name>blocker</name> <operator>&amp;&amp;</operator> <call><name>value</name><argument_list>(<argument><expr><name>first</name></expr></argument>)</argument_list></call> <operator>==</operator> <name>l_True</name></expr>)</condition><then><block>{
                <expr_stmt><expr><operator>*</operator><name>j</name><operator>++</operator> <operator>=</operator> <name>w</name></expr>;</expr_stmt> <continue>continue;</continue> }</block></then></if>

            <comment type="line">// Look for new watch:</comment>
            <for>for <control>(<init><decl><type><name>int</name></type> <name>k</name> <init>= <expr><literal type="number">2</literal></expr></init></decl>;</init> <condition><expr><name>k</name> <operator>&lt;</operator> <call><name><name>c</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>k</name><operator>++</operator></expr></incr>)</control>
                <block type="pseudo"><if>if <condition>(<expr><call><name>value</name><argument_list>(<argument><expr><name><name>c</name><index>[<expr><name>k</name></expr>]</index></name></expr></argument>)</argument_list></call> <operator>!=</operator> <name>l_False</name></expr>)</condition><then><block>{
                    <expr_stmt><expr><name><name>c</name><index>[<expr><literal type="number">1</literal></expr>]</index></name> <operator>=</operator> <name><name>c</name><index>[<expr><name>k</name></expr>]</index></name></expr>;</expr_stmt> <expr_stmt><expr><name><name>c</name><index>[<expr><name>k</name></expr>]</index></name> <operator>=</operator> <name>false_lit</name></expr>;</expr_stmt>
                    <expr_stmt><expr><name><name>watches</name><index>[<expr><operator>~</operator><name><name>c</name><index>[<expr><literal type="number">1</literal></expr>]</index></name></expr>]</index></name><operator>.</operator><call><name>push</name><argument_list>(<argument><expr><name>w</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
                    <goto>goto <name>NextClause</name>;</goto> }</block></then></if></block></for>

            <comment type="line">// Did not find watch -- clause is unit under assignment:</comment>
            <expr_stmt><expr><operator>*</operator><name>j</name><operator>++</operator> <operator>=</operator> <name>w</name></expr>;</expr_stmt>
            <if>if <condition>(<expr><call><name>value</name><argument_list>(<argument><expr><name>first</name></expr></argument>)</argument_list></call> <operator>==</operator> <name>l_False</name></expr>)</condition><then><block>{
                <expr_stmt><expr><name>confl</name> <operator>=</operator> <name>cr</name></expr>;</expr_stmt>
                <expr_stmt><expr><name>qhead</name> <operator>=</operator> <call><name><name>trail</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</expr_stmt>
                <comment type="line">// Copy the remaining watches:</comment>
                <while>while <condition>(<expr><name>i</name> <operator>&lt;</operator> <name>end</name></expr>)</condition>
                    <block type="pseudo"><expr_stmt><expr><operator>*</operator><name>j</name><operator>++</operator> <operator>=</operator> <operator>*</operator><name>i</name><operator>++</operator></expr>;</expr_stmt></block></while>
            }</block></then><else>else
                <block type="pseudo"><expr_stmt><expr><call><name>uncheckedEnqueue</name><argument_list>(<argument><expr><name>first</name></expr></argument>, <argument><expr><name>cr</name></expr></argument>)</argument_list></call></expr>;</expr_stmt></block></else></if>

        <label><name>NextClause</name>:</label><empty_stmt>;</empty_stmt>
        }</block></for>
        <expr_stmt><expr><call><name><name>ws</name><operator>.</operator><name>shrink</name></name><argument_list>(<argument><expr><name>i</name> <operator>-</operator> <name>j</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    }</block></while>
    <expr_stmt><expr><name>propagations</name> <operator>+=</operator> <name>num_props</name></expr>;</expr_stmt>
    <expr_stmt><expr><name>simpDB_props</name> <operator>-=</operator> <name>num_props</name></expr>;</expr_stmt>

    <return>return <expr><name>confl</name></expr>;</return>
}</block></function>


<comment type="block">/*_________________________________________________________________________________________________
|
|  reduceDB : ()  -&gt;  [void]
|  
|  Description:
|    Remove half of the learnt clauses, minus the clauses locked by the current assignment. Locked
|    clauses are clauses that are reason to some assignment. Binary clauses are never removed.
|________________________________________________________________________________________________@*/</comment>
<struct>struct <name>reduceDB_lt</name> <block>{<public type="default"> 
    <decl_stmt><decl><type><name>ClauseAllocator</name><modifier>&amp;</modifier></type> <name>ca</name></decl>;</decl_stmt>
    <constructor><name>reduceDB_lt</name><parameter_list>(<parameter><decl><type><name>ClauseAllocator</name><modifier>&amp;</modifier></type> <name>ca_</name></decl></parameter>)</parameter_list> <member_init_list>: <call><name>ca</name><argument_list>(<argument><expr><name>ca_</name></expr></argument>)</argument_list></call> </member_init_list><block>{}</block></constructor>
    <function type="operator"><type><name>bool</name></type> <name>operator <name>()</name></name> <parameter_list>(<parameter><decl><type><name>CRef</name></type> <name>x</name></decl></parameter>, <parameter><decl><type><name>CRef</name></type> <name>y</name></decl></parameter>)</parameter_list> <block>{ 
        <return>return <expr><name><name>ca</name><index>[<expr><name>x</name></expr>]</index></name><operator>.</operator><call><name>size</name><argument_list>()</argument_list></call> <operator>&gt;</operator> <literal type="number">2</literal> <operator>&amp;&amp;</operator> <operator>(</operator><name><name>ca</name><index>[<expr><name>y</name></expr>]</index></name><operator>.</operator><call><name>size</name><argument_list>()</argument_list></call> <operator>==</operator> <literal type="number">2</literal> <operator>||</operator> <name><name>ca</name><index>[<expr><name>x</name></expr>]</index></name><operator>.</operator><call><name>activity</name><argument_list>()</argument_list></call> <operator>&lt;</operator> <name><name>ca</name><index>[<expr><name>y</name></expr>]</index></name><operator>.</operator><call><name>activity</name><argument_list>()</argument_list></call><operator>)</operator></expr>;</return> }</block></function> 
</public>}</block>;</struct>
<function><type><name>void</name></type> <name><name>Solver</name><operator>::</operator><name>reduceDB</name></name><parameter_list>()</parameter_list>
<block>{
    <decl_stmt><decl><type><name>int</name></type>     <name>i</name></decl>, <decl><type ref="prev"/><name>j</name></decl>;</decl_stmt>
    <decl_stmt><decl><type><name>double</name></type>  <name>extra_lim</name> <init>= <expr><name>cla_inc</name> <operator>/</operator> <call><name><name>learnts</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr></init></decl>;</decl_stmt>    <comment type="line">// Remove any clause below this activity</comment>

    <expr_stmt><expr><call><name>sort</name><argument_list>(<argument><expr><name>learnts</name></expr></argument>, <argument><expr><call><name>reduceDB_lt</name><argument_list>(<argument><expr><name>ca</name></expr></argument>)</argument_list></call></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <comment type="line">// Don't delete binary or locked clauses. From the rest, delete clauses from the first half</comment>
    <comment type="line">// and clauses with activity smaller than 'extra_lim':</comment>
    <for>for <control>(<init><expr><name>i</name> <operator>=</operator> <name>j</name> <operator>=</operator> <literal type="number">0</literal></expr>;</init> <condition><expr><name>i</name> <operator>&lt;</operator> <call><name><name>learnts</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>i</name><operator>++</operator></expr></incr>)</control><block>{
        <decl_stmt><decl><type><name>Clause</name><modifier>&amp;</modifier></type> <name>c</name> <init>= <expr><name><name>ca</name><index>[<expr><name><name>learnts</name><index>[<expr><name>i</name></expr>]</index></name></expr>]</index></name></expr></init></decl>;</decl_stmt>
        <if>if <condition>(<expr><call><name><name>c</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call> <operator>&gt;</operator> <literal type="number">2</literal> <operator>&amp;&amp;</operator> <operator>!</operator><call><name>locked</name><argument_list>(<argument><expr><name>c</name></expr></argument>)</argument_list></call> <operator>&amp;&amp;</operator> <operator>(</operator><name>i</name> <operator>&lt;</operator> <call><name><name>learnts</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call> <operator>/</operator> <literal type="number">2</literal> <operator>||</operator> <call><name><name>c</name><operator>.</operator><name>activity</name></name><argument_list>()</argument_list></call> <operator>&lt;</operator> <name>extra_lim</name><operator>)</operator></expr>)</condition><then>
            <block type="pseudo"><expr_stmt><expr><call><name>removeClause</name><argument_list>(<argument><expr><name><name>learnts</name><index>[<expr><name>i</name></expr>]</index></name></expr></argument>)</argument_list></call></expr>;</expr_stmt></block></then>
        <else>else
            <block type="pseudo"><expr_stmt><expr><name><name>learnts</name><index>[<expr><name>j</name><operator>++</operator></expr>]</index></name> <operator>=</operator> <name><name>learnts</name><index>[<expr><name>i</name></expr>]</index></name></expr>;</expr_stmt></block></else></if>
    }</block></for>
    <expr_stmt><expr><call><name><name>learnts</name><operator>.</operator><name>shrink</name></name><argument_list>(<argument><expr><name>i</name> <operator>-</operator> <name>j</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <expr_stmt><expr><call><name>checkGarbage</name><argument_list>()</argument_list></call></expr>;</expr_stmt>
}</block></function>


<function><type><name>void</name></type> <name><name>Solver</name><operator>::</operator><name>removeSatisfied</name></name><parameter_list>(<parameter><decl><type><name><name>vec</name><argument_list type="generic">&lt;<argument><expr><name>CRef</name></expr></argument>&gt;</argument_list></name><modifier>&amp;</modifier></type> <name>cs</name></decl></parameter>)</parameter_list>
<block>{
    <decl_stmt><decl><type><name>int</name></type> <name>i</name></decl>, <decl><type ref="prev"/><name>j</name></decl>;</decl_stmt>
    <for>for <control>(<init><expr><name>i</name> <operator>=</operator> <name>j</name> <operator>=</operator> <literal type="number">0</literal></expr>;</init> <condition><expr><name>i</name> <operator>&lt;</operator> <call><name><name>cs</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>i</name><operator>++</operator></expr></incr>)</control><block>{
        <decl_stmt><decl><type><name>Clause</name><modifier>&amp;</modifier></type> <name>c</name> <init>= <expr><name><name>ca</name><index>[<expr><name><name>cs</name><index>[<expr><name>i</name></expr>]</index></name></expr>]</index></name></expr></init></decl>;</decl_stmt>
        <if>if <condition>(<expr><call><name>satisfied</name><argument_list>(<argument><expr><name>c</name></expr></argument>)</argument_list></call></expr>)</condition><then>
            <block type="pseudo"><expr_stmt><expr><call><name>removeClause</name><argument_list>(<argument><expr><name><name>cs</name><index>[<expr><name>i</name></expr>]</index></name></expr></argument>)</argument_list></call></expr>;</expr_stmt></block></then>
        <else>else
            <block type="pseudo"><expr_stmt><expr><name><name>cs</name><index>[<expr><name>j</name><operator>++</operator></expr>]</index></name> <operator>=</operator> <name><name>cs</name><index>[<expr><name>i</name></expr>]</index></name></expr>;</expr_stmt></block></else></if>
    }</block></for>
    <expr_stmt><expr><call><name><name>cs</name><operator>.</operator><name>shrink</name></name><argument_list>(<argument><expr><name>i</name> <operator>-</operator> <name>j</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
}</block></function>


<function><type><name>void</name></type> <name><name>Solver</name><operator>::</operator><name>rebuildOrderHeap</name></name><parameter_list>()</parameter_list>
<block>{
    <decl_stmt><decl><type><name><name>vec</name><argument_list type="generic">&lt;<argument><expr><name>Var</name></expr></argument>&gt;</argument_list></name></type> <name>vs</name></decl>;</decl_stmt>
    <for>for <control>(<init><decl><type><name>Var</name></type> <name>v</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</init> <condition><expr><name>v</name> <operator>&lt;</operator> <call><name>nVars</name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>v</name><operator>++</operator></expr></incr>)</control>
        <block type="pseudo"><if>if <condition>(<expr><name><name>decision</name><index>[<expr><name>v</name></expr>]</index></name> <operator>&amp;&amp;</operator> <call><name>value</name><argument_list>(<argument><expr><name>v</name></expr></argument>)</argument_list></call> <operator>==</operator> <name>l_Undef</name></expr>)</condition><then>
            <block type="pseudo"><expr_stmt><expr><call><name><name>vs</name><operator>.</operator><name>push</name></name><argument_list>(<argument><expr><name>v</name></expr></argument>)</argument_list></call></expr>;</expr_stmt></block></then></if></block></for>
    <expr_stmt><expr><call><name><name>order_heap</name><operator>.</operator><name>build</name></name><argument_list>(<argument><expr><name>vs</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
}</block></function>


<comment type="block">/*_________________________________________________________________________________________________
|
|  simplify : [void]  -&gt;  [bool]
|  
|  Description:
|    Simplify the clause database according to the current top-level assigment. Currently, the only
|    thing done here is the removal of satisfied clauses, but more things can be put here.
|________________________________________________________________________________________________@*/</comment>
<function><type><name>bool</name></type> <name><name>Solver</name><operator>::</operator><name>simplify</name></name><parameter_list>()</parameter_list>
<block>{
    <expr_stmt><expr><call><name>assert</name><argument_list>(<argument><expr><call><name>decisionLevel</name><argument_list>()</argument_list></call> <operator>==</operator> <literal type="number">0</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt>

    <if>if <condition>(<expr><operator>!</operator><name>ok</name> <operator>||</operator> <call><name>propagate</name><argument_list>()</argument_list></call> <operator>!=</operator> <name>CRef_Undef</name></expr>)</condition><then>
        <block type="pseudo"><return>return <expr><name>ok</name> <operator>=</operator> <literal type="boolean">false</literal></expr>;</return></block></then></if>

    <if>if <condition>(<expr><call><name>nAssigns</name><argument_list>()</argument_list></call> <operator>==</operator> <name>simpDB_assigns</name> <operator>||</operator> <operator>(</operator><name>simpDB_props</name> <operator>&gt;</operator> <literal type="number">0</literal><operator>)</operator></expr>)</condition><then>
        <block type="pseudo"><return>return <expr><literal type="boolean">true</literal></expr>;</return></block></then></if>

    <comment type="line">// Remove satisfied clauses:</comment>
    <expr_stmt><expr><call><name>removeSatisfied</name><argument_list>(<argument><expr><name>learnts</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <if>if <condition>(<expr><name>remove_satisfied</name></expr>)</condition><then>        <comment type="line">// Can be turned off.</comment>
        <block type="pseudo"><expr_stmt><expr><call><name>removeSatisfied</name><argument_list>(<argument><expr><name>clauses</name></expr></argument>)</argument_list></call></expr>;</expr_stmt></block></then></if>
    <expr_stmt><expr><call><name>checkGarbage</name><argument_list>()</argument_list></call></expr>;</expr_stmt>
    <expr_stmt><expr><call><name>rebuildOrderHeap</name><argument_list>()</argument_list></call></expr>;</expr_stmt>

    <expr_stmt><expr><name>simpDB_assigns</name> <operator>=</operator> <call><name>nAssigns</name><argument_list>()</argument_list></call></expr>;</expr_stmt>
    <expr_stmt><expr><name>simpDB_props</name>   <operator>=</operator> <name>clauses_literals</name> <operator>+</operator> <name>learnts_literals</name></expr>;</expr_stmt>   <comment type="line">// (shouldn't depend on stats really, but it will do for now)</comment>

    <return>return <expr><literal type="boolean">true</literal></expr>;</return>
}</block></function>


<comment type="block">/*_________________________________________________________________________________________________
|
|  search : (nof_conflicts : int) (params : const SearchParams&amp;)  -&gt;  [lbool]
|  
|  Description:
|    Search for a model the specified number of conflicts. 
|    NOTE! Use negative value for 'nof_conflicts' indicate infinity.
|  
|  Output:
|    'l_True' if a partial assigment that is consistent with respect to the clauseset is found. If
|    all variables are decision variables, this means that the clause set is satisfiable. 'l_False'
|    if the clause set is unsatisfiable. 'l_Undef' if the bound on number of conflicts is reached.
|________________________________________________________________________________________________@*/</comment>
<function><type><name>lbool</name></type> <name><name>Solver</name><operator>::</operator><name>search</name></name><parameter_list>(<parameter><decl><type><name>int</name></type> <name>nof_conflicts</name></decl></parameter>)</parameter_list>
<block>{
    <expr_stmt><expr><call><name>assert</name><argument_list>(<argument><expr><name>ok</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <decl_stmt><decl><type><name>int</name></type>         <name>backtrack_level</name></decl>;</decl_stmt>
    <decl_stmt><decl><type><name>int</name></type>         <name>conflictC</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</decl_stmt>
    <decl_stmt><decl><type><name><name>vec</name><argument_list type="generic">&lt;<argument><expr><name>Lit</name></expr></argument>&gt;</argument_list></name></type>    <name>learnt_clause</name></decl>;</decl_stmt>
    <expr_stmt><expr><name>starts</name><operator>++</operator></expr>;</expr_stmt>

    <for>for <control>(<init>;</init><condition>;</condition><incr/>)</control><block>{
        <decl_stmt><decl><type><name>CRef</name></type> <name>confl</name> <init>= <expr><call><name>propagate</name><argument_list>()</argument_list></call></expr></init></decl>;</decl_stmt>
        <if>if <condition>(<expr><name>confl</name> <operator>!=</operator> <name>CRef_Undef</name></expr>)</condition><then><block>{
            <comment type="line">// CONFLICT</comment>
            <expr_stmt><expr><name>conflicts</name><operator>++</operator></expr>;</expr_stmt> <expr_stmt><expr><name>conflictC</name><operator>++</operator></expr>;</expr_stmt>
            <if>if <condition>(<expr><call><name>decisionLevel</name><argument_list>()</argument_list></call> <operator>==</operator> <literal type="number">0</literal></expr>)</condition><then> <block type="pseudo"><return>return <expr><name>l_False</name></expr>;</return></block></then></if>

            <expr_stmt><expr><call><name><name>learnt_clause</name><operator>.</operator><name>clear</name></name><argument_list>()</argument_list></call></expr>;</expr_stmt>
            <expr_stmt><expr><call><name>analyze</name><argument_list>(<argument><expr><name>confl</name></expr></argument>, <argument><expr><name>learnt_clause</name></expr></argument>, <argument><expr><name>backtrack_level</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
            <expr_stmt><expr><call><name>cancelUntil</name><argument_list>(<argument><expr><name>backtrack_level</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>

            <if>if <condition>(<expr><call><name><name>learnt_clause</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call> <operator>==</operator> <literal type="number">1</literal></expr>)</condition><then><block>{
                <expr_stmt><expr><call><name>uncheckedEnqueue</name><argument_list>(<argument><expr><name><name>learnt_clause</name><index>[<expr><literal type="number">0</literal></expr>]</index></name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
            }</block></then><else>else<block>{
                <decl_stmt><decl><type><name>CRef</name></type> <name>cr</name> <init>= <expr><call><name><name>ca</name><operator>.</operator><name>alloc</name></name><argument_list>(<argument><expr><name>learnt_clause</name></expr></argument>, <argument><expr><literal type="boolean">true</literal></expr></argument>)</argument_list></call></expr></init></decl>;</decl_stmt>
                <expr_stmt><expr><call><name><name>learnts</name><operator>.</operator><name>push</name></name><argument_list>(<argument><expr><name>cr</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
                <expr_stmt><expr><call><name>attachClause</name><argument_list>(<argument><expr><name>cr</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
                <expr_stmt><expr><call><name>claBumpActivity</name><argument_list>(<argument><expr><name><name>ca</name><index>[<expr><name>cr</name></expr>]</index></name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
                <expr_stmt><expr><call><name>uncheckedEnqueue</name><argument_list>(<argument><expr><name><name>learnt_clause</name><index>[<expr><literal type="number">0</literal></expr>]</index></name></expr></argument>, <argument><expr><name>cr</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
            }</block></else></if>

            <expr_stmt><expr><call><name>varDecayActivity</name><argument_list>()</argument_list></call></expr>;</expr_stmt>
            <expr_stmt><expr><call><name>claDecayActivity</name><argument_list>()</argument_list></call></expr>;</expr_stmt>

            <if>if <condition>(<expr><operator>--</operator><name>learntsize_adjust_cnt</name> <operator>==</operator> <literal type="number">0</literal></expr>)</condition><then><block>{
                <expr_stmt><expr><name>learntsize_adjust_confl</name> <operator>*=</operator> <name>learntsize_adjust_inc</name></expr>;</expr_stmt>
                <expr_stmt><expr><name>learntsize_adjust_cnt</name>    <operator>=</operator> <operator>(</operator><name>int</name><operator>)</operator><name>learntsize_adjust_confl</name></expr>;</expr_stmt>
                <expr_stmt><expr><name>max_learnts</name>             <operator>*=</operator> <name>learntsize_inc</name></expr>;</expr_stmt>

                <if>if <condition>(<expr><name>verbosity</name> <operator>&gt;=</operator> <literal type="number">1</literal></expr>)</condition><then>
                    <block type="pseudo"><expr_stmt><expr><call><name>printf</name><argument_list>(<argument><expr><literal type="string">"| %9d | %7d %8d %8d | %8d %8d %6.0f | %6.3f %% |\n"</literal></expr></argument>, 
                           <argument><expr><operator>(</operator><name>int</name><operator>)</operator><name>conflicts</name></expr></argument>, 
                           <argument><expr><operator>(</operator><name>int</name><operator>)</operator><name>dec_vars</name> <operator>-</operator> <operator>(</operator><ternary><condition><expr><call><name><name>trail_lim</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call> <operator>==</operator> <literal type="number">0</literal></expr> ?</condition><then> <expr><call><name><name>trail</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr> </then><else>: <expr><name><name>trail_lim</name><index>[<expr><literal type="number">0</literal></expr>]</index></name></expr></else></ternary><operator>)</operator></expr></argument>, <argument><expr><call><name>nClauses</name><argument_list>()</argument_list></call></expr></argument>, <argument><expr><operator>(</operator><name>int</name><operator>)</operator><name>clauses_literals</name></expr></argument>, 
                           <argument><expr><operator>(</operator><name>int</name><operator>)</operator><name>max_learnts</name></expr></argument>, <argument><expr><call><name>nLearnts</name><argument_list>()</argument_list></call></expr></argument>, <argument><expr><operator>(</operator><name>double</name><operator>)</operator><name>learnts_literals</name><operator>/</operator><call><name>nLearnts</name><argument_list>()</argument_list></call></expr></argument>, <argument><expr><call><name>progressEstimate</name><argument_list>()</argument_list></call><operator>*</operator><literal type="number">100</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt></block></then></if>
            }</block></then></if>

        }</block></then><else>else<block>{
            <comment type="line">// NO CONFLICT</comment>
            <if>if <condition>(<expr><name>nof_conflicts</name> <operator>&gt;=</operator> <literal type="number">0</literal> <operator>&amp;&amp;</operator> <name>conflictC</name> <operator>&gt;=</operator> <name>nof_conflicts</name> <operator>||</operator> <operator>!</operator><call><name>withinBudget</name><argument_list>()</argument_list></call></expr>)</condition><then><block>{
                <comment type="line">// Reached bound on number of conflicts:</comment>
                <expr_stmt><expr><name>progress_estimate</name> <operator>=</operator> <call><name>progressEstimate</name><argument_list>()</argument_list></call></expr>;</expr_stmt>
                <expr_stmt><expr><call><name>cancelUntil</name><argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt>
                <return>return <expr><name>l_Undef</name></expr>;</return> }</block></then></if>

            <comment type="line">// Simplify the set of problem clauses:</comment>
            <if>if <condition>(<expr><call><name>decisionLevel</name><argument_list>()</argument_list></call> <operator>==</operator> <literal type="number">0</literal> <operator>&amp;&amp;</operator> <operator>!</operator><call><name>simplify</name><argument_list>()</argument_list></call></expr>)</condition><then>
                <block type="pseudo"><return>return <expr><name>l_False</name></expr>;</return></block></then></if>

            <if>if <condition>(<expr><call><name><name>learnts</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call><operator>-</operator><call><name>nAssigns</name><argument_list>()</argument_list></call> <operator>&gt;=</operator> <name>max_learnts</name></expr>)</condition><then>
                <comment type="line">// Reduce the set of learnt clauses:</comment>
                <block type="pseudo"><expr_stmt><expr><call><name>reduceDB</name><argument_list>()</argument_list></call></expr>;</expr_stmt></block></then></if>

            <decl_stmt><decl><type><name>Lit</name></type> <name>next</name> <init>= <expr><name>lit_Undef</name></expr></init></decl>;</decl_stmt>
            <while>while <condition>(<expr><call><name>decisionLevel</name><argument_list>()</argument_list></call> <operator>&lt;</operator> <call><name><name>assumptions</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>)</condition><block>{
                <comment type="line">// Perform user provided assumption:</comment>
                <decl_stmt><decl><type><name>Lit</name></type> <name>p</name> <init>= <expr><name><name>assumptions</name><index>[<expr><call><name>decisionLevel</name><argument_list>()</argument_list></call></expr>]</index></name></expr></init></decl>;</decl_stmt>
                <if>if <condition>(<expr><call><name>value</name><argument_list>(<argument><expr><name>p</name></expr></argument>)</argument_list></call> <operator>==</operator> <name>l_True</name></expr>)</condition><then><block>{
                    <comment type="line">// Dummy decision level:</comment>
                    <expr_stmt><expr><call><name>newDecisionLevel</name><argument_list>()</argument_list></call></expr>;</expr_stmt>
                }</block></then><elseif>else <if>if <condition>(<expr><call><name>value</name><argument_list>(<argument><expr><name>p</name></expr></argument>)</argument_list></call> <operator>==</operator> <name>l_False</name></expr>)</condition><then><block>{
                    <expr_stmt><expr><call><name>analyzeFinal</name><argument_list>(<argument><expr><operator>~</operator><name>p</name></expr></argument>, <argument><expr><name>conflict</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
                    <return>return <expr><name>l_False</name></expr>;</return>
                }</block></then></if></elseif><else>else<block>{
                    <expr_stmt><expr><name>next</name> <operator>=</operator> <name>p</name></expr>;</expr_stmt>
                    <break>break;</break>
                }</block></else></if>
            }</block></while>

            <if>if <condition>(<expr><name>next</name> <operator>==</operator> <name>lit_Undef</name></expr>)</condition><then><block>{
                <comment type="line">// New variable decision:</comment>
                <expr_stmt><expr><name>decisions</name><operator>++</operator></expr>;</expr_stmt>
                <expr_stmt><expr><name>next</name> <operator>=</operator> <call><name>pickBranchLit</name><argument_list>()</argument_list></call></expr>;</expr_stmt>

                <if>if <condition>(<expr><name>next</name> <operator>==</operator> <name>lit_Undef</name></expr>)</condition><then>
                    <comment type="line">// Model found:</comment>
                    <block type="pseudo"><return>return <expr><name>l_True</name></expr>;</return></block></then></if>
            }</block></then></if>

            <comment type="line">// Increase decision level and enqueue 'next'</comment>
            <expr_stmt><expr><call><name>newDecisionLevel</name><argument_list>()</argument_list></call></expr>;</expr_stmt>
            <expr_stmt><expr><call><name>uncheckedEnqueue</name><argument_list>(<argument><expr><name>next</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
        }</block></else></if>
    }</block></for>
}</block></function>


<function><type><name>double</name></type> <name><name>Solver</name><operator>::</operator><name>progressEstimate</name></name><parameter_list>()</parameter_list> <specifier>const</specifier>
<block>{
    <decl_stmt><decl><type><name>double</name></type>  <name>progress</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</decl_stmt>
    <decl_stmt><decl><type><name>double</name></type>  <name>F</name> <init>= <expr><literal type="number">1.0</literal> <operator>/</operator> <call><name>nVars</name><argument_list>()</argument_list></call></expr></init></decl>;</decl_stmt>

    <for>for <control>(<init><decl><type><name>int</name></type> <name>i</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</init> <condition><expr><name>i</name> <operator>&lt;=</operator> <call><name>decisionLevel</name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>i</name><operator>++</operator></expr></incr>)</control><block>{
        <decl_stmt><decl><type><name>int</name></type> <name>beg</name> <init>= <expr><ternary><condition><expr><name>i</name> <operator>==</operator> <literal type="number">0</literal></expr> ?</condition><then> <expr><literal type="number">0</literal></expr> </then><else>: <expr><name><name>trail_lim</name><index>[<expr><name>i</name> <operator>-</operator> <literal type="number">1</literal></expr>]</index></name></expr></else></ternary></expr></init></decl>;</decl_stmt>
        <decl_stmt><decl><type><name>int</name></type> <name>end</name> <init>= <expr><ternary><condition><expr><name>i</name> <operator>==</operator> <call><name>decisionLevel</name><argument_list>()</argument_list></call></expr> ?</condition><then> <expr><call><name><name>trail</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr> </then><else>: <expr><name><name>trail_lim</name><index>[<expr><name>i</name></expr>]</index></name></expr></else></ternary></expr></init></decl>;</decl_stmt>
        <expr_stmt><expr><name>progress</name> <operator>+=</operator> <call><name>pow</name><argument_list>(<argument><expr><name>F</name></expr></argument>, <argument><expr><name>i</name></expr></argument>)</argument_list></call> <operator>*</operator> <operator>(</operator><name>end</name> <operator>-</operator> <name>beg</name><operator>)</operator></expr>;</expr_stmt>
    }</block></for>

    <return>return <expr><name>progress</name> <operator>/</operator> <call><name>nVars</name><argument_list>()</argument_list></call></expr>;</return>
}</block></function>

<comment type="block">/*
  Finite subsequences of the Luby-sequence:

  0: 1
  1: 1 1 2
  2: 1 1 2 1 1 2 4
  3: 1 1 2 1 1 2 4 1 1 2 1 1 2 4 8
  ...


 */</comment>

<function><specifier>static</specifier> <type><name>double</name></type> <name>luby</name><parameter_list>(<parameter><decl><type><name>double</name></type> <name>y</name></decl></parameter>, <parameter><decl><type><name>int</name></type> <name>x</name></decl></parameter>)</parameter_list><block>{

    <comment type="line">// Find the finite subsequence that contains index 'x', and the</comment>
    <comment type="line">// size of that subsequence:</comment>
    <decl_stmt><decl><type><name>int</name></type> <name>size</name></decl>, <decl><type ref="prev"/><name>seq</name></decl>;</decl_stmt>
    <for>for <control>(<init><expr><name>size</name> <operator>=</operator> <literal type="number">1</literal></expr><operator>,</operator> <expr><name>seq</name> <operator>=</operator> <literal type="number">0</literal></expr>;</init> <condition><expr><name>size</name> <operator>&lt;</operator> <name>x</name><operator>+</operator><literal type="number">1</literal></expr>;</condition> <incr><expr><name>seq</name><operator>++</operator></expr><operator>,</operator> <expr><name>size</name> <operator>=</operator> <literal type="number">2</literal><operator>*</operator><name>size</name><operator>+</operator><literal type="number">1</literal></expr></incr>)</control><block type="pseudo"><empty_stmt>;</empty_stmt></block></for>

    <while>while <condition>(<expr><name>size</name><operator>-</operator><literal type="number">1</literal> <operator>!=</operator> <name>x</name></expr>)</condition><block>{
        <expr_stmt><expr><name>size</name> <operator>=</operator> <operator>(</operator><name>size</name><operator>-</operator><literal type="number">1</literal><operator>)</operator><operator>&gt;&gt;</operator><literal type="number">1</literal></expr>;</expr_stmt>
        <expr_stmt><expr><name>seq</name><operator>--</operator></expr>;</expr_stmt>
        <expr_stmt><expr><name>x</name> <operator>=</operator> <name>x</name> <operator>%</operator> <name>size</name></expr>;</expr_stmt>
    }</block></while>

    <return>return <expr><call><name>pow</name><argument_list>(<argument><expr><name>y</name></expr></argument>, <argument><expr><name>seq</name></expr></argument>)</argument_list></call></expr>;</return>
}</block></function>

<comment type="line">// NOTE: assumptions passed in member-variable 'assumptions'.</comment>
<function><type><name>lbool</name></type> <name><name>Solver</name><operator>::</operator><name>solve_</name></name><parameter_list>()</parameter_list>
<block>{
    <expr_stmt><expr><call><name><name>model</name><operator>.</operator><name>clear</name></name><argument_list>()</argument_list></call></expr>;</expr_stmt>
    <expr_stmt><expr><call><name><name>conflict</name><operator>.</operator><name>clear</name></name><argument_list>()</argument_list></call></expr>;</expr_stmt>
    <if>if <condition>(<expr><operator>!</operator><name>ok</name></expr>)</condition><then> <block type="pseudo"><return>return <expr><name>l_False</name></expr>;</return></block></then></if>

    <expr_stmt><expr><name>solves</name><operator>++</operator></expr>;</expr_stmt>

    <expr_stmt><expr><name>max_learnts</name>               <operator>=</operator> <call><name>nClauses</name><argument_list>()</argument_list></call> <operator>*</operator> <name>learntsize_factor</name></expr>;</expr_stmt>
    <expr_stmt><expr><name>learntsize_adjust_confl</name>   <operator>=</operator> <name>learntsize_adjust_start_confl</name></expr>;</expr_stmt>
    <expr_stmt><expr><name>learntsize_adjust_cnt</name>     <operator>=</operator> <operator>(</operator><name>int</name><operator>)</operator><name>learntsize_adjust_confl</name></expr>;</expr_stmt>
    <decl_stmt><decl><type><name>lbool</name></type>   <name>status</name>            <init>= <expr><name>l_Undef</name></expr></init></decl>;</decl_stmt>

    <if>if <condition>(<expr><name>verbosity</name> <operator>&gt;=</operator> <literal type="number">1</literal></expr>)</condition><then><block>{
        <expr_stmt><expr><call><name>printf</name><argument_list>(<argument><expr><literal type="string">"============================[ Search Statistics ]==============================\n"</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt>
        <expr_stmt><expr><call><name>printf</name><argument_list>(<argument><expr><literal type="string">"| Conflicts |          ORIGINAL         |          LEARNT          | Progress |\n"</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt>
        <expr_stmt><expr><call><name>printf</name><argument_list>(<argument><expr><literal type="string">"|           |    Vars  Clauses Literals |    Limit  Clauses Lit/Cl |          |\n"</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt>
        <expr_stmt><expr><call><name>printf</name><argument_list>(<argument><expr><literal type="string">"===============================================================================\n"</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    }</block></then></if>

    <comment type="line">// Search:</comment>
    <decl_stmt><decl><type><name>int</name></type> <name>curr_restarts</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</decl_stmt>
    <while>while <condition>(<expr><name>status</name> <operator>==</operator> <name>l_Undef</name></expr>)</condition><block>{
        <decl_stmt><decl><type><name>double</name></type> <name>rest_base</name> <init>= <expr><ternary><condition><expr><name>luby_restart</name></expr> ?</condition><then> <expr><call><name>luby</name><argument_list>(<argument><expr><name>restart_inc</name></expr></argument>, <argument><expr><name>curr_restarts</name></expr></argument>)</argument_list></call></expr> </then><else>: <expr><call><name>pow</name><argument_list>(<argument><expr><name>restart_inc</name></expr></argument>, <argument><expr><name>curr_restarts</name></expr></argument>)</argument_list></call></expr></else></ternary></expr></init></decl>;</decl_stmt>
        <expr_stmt><expr><name>status</name> <operator>=</operator> <call><name>search</name><argument_list>(<argument><expr><name>rest_base</name> <operator>*</operator> <name>restart_first</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
        <if>if <condition>(<expr><operator>!</operator><call><name>withinBudget</name><argument_list>()</argument_list></call></expr>)</condition><then> <block type="pseudo"><break>break;</break></block></then></if>
        <expr_stmt><expr><name>curr_restarts</name><operator>++</operator></expr>;</expr_stmt>
    }</block></while>

    <if>if <condition>(<expr><name>verbosity</name> <operator>&gt;=</operator> <literal type="number">1</literal></expr>)</condition><then>
        <block type="pseudo"><expr_stmt><expr><call><name>printf</name><argument_list>(<argument><expr><literal type="string">"===============================================================================\n"</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt></block></then></if>


    <if>if <condition>(<expr><name>status</name> <operator>==</operator> <name>l_True</name></expr>)</condition><then><block>{
        <comment type="line">// Extend &amp; copy model:</comment>
        <expr_stmt><expr><call><name><name>model</name><operator>.</operator><name>growTo</name></name><argument_list>(<argument><expr><call><name>nVars</name><argument_list>()</argument_list></call></expr></argument>)</argument_list></call></expr>;</expr_stmt>
        <for>for <control>(<init><decl><type><name>int</name></type> <name>i</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</init> <condition><expr><name>i</name> <operator>&lt;</operator> <call><name>nVars</name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>i</name><operator>++</operator></expr></incr>)</control> <block type="pseudo"><expr_stmt><expr><name><name>model</name><index>[<expr><name>i</name></expr>]</index></name> <operator>=</operator> <call><name>value</name><argument_list>(<argument><expr><name>i</name></expr></argument>)</argument_list></call></expr>;</expr_stmt></block></for>
    }</block></then><elseif>else <if>if <condition>(<expr><name>status</name> <operator>==</operator> <name>l_False</name> <operator>&amp;&amp;</operator> <call><name><name>conflict</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call> <operator>==</operator> <literal type="number">0</literal></expr>)</condition><then>
        <block type="pseudo"><expr_stmt><expr><name>ok</name> <operator>=</operator> <literal type="boolean">false</literal></expr>;</expr_stmt></block></then></if></elseif></if>

    <expr_stmt><expr><call><name>cancelUntil</name><argument_list>(<argument><expr><literal type="number">0</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <return>return <expr><name>status</name></expr>;</return>
}</block></function>

<comment type="line">//=================================================================================================</comment>
<comment type="line">// Writing CNF to DIMACS:</comment>
<comment type="line">// </comment>
<comment type="line">// FIXME: this needs to be rewritten completely.</comment>

<function><specifier>static</specifier> <type><name>Var</name></type> <name>mapVar</name><parameter_list>(<parameter><decl><type><name>Var</name></type> <name>x</name></decl></parameter>, <parameter><decl><type><name><name>vec</name><argument_list type="generic">&lt;<argument><expr><name>Var</name></expr></argument>&gt;</argument_list></name><modifier>&amp;</modifier></type> <name>map</name></decl></parameter>, <parameter><decl><type><name>Var</name><modifier>&amp;</modifier></type> <name>max</name></decl></parameter>)</parameter_list>
<block>{
    <if>if <condition>(<expr><call><name><name>map</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call> <operator>&lt;=</operator> <name>x</name> <operator>||</operator> <name><name>map</name><index>[<expr><name>x</name></expr>]</index></name> <operator>==</operator> <operator>-</operator><literal type="number">1</literal></expr>)</condition><then><block>{
        <expr_stmt><expr><call><name><name>map</name><operator>.</operator><name>growTo</name></name><argument_list>(<argument><expr><name>x</name><operator>+</operator><literal type="number">1</literal></expr></argument>, <argument><expr><operator>-</operator><literal type="number">1</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt>
        <expr_stmt><expr><name><name>map</name><index>[<expr><name>x</name></expr>]</index></name> <operator>=</operator> <name>max</name><operator>++</operator></expr>;</expr_stmt>
    }</block></then></if>
    <return>return <expr><name><name>map</name><index>[<expr><name>x</name></expr>]</index></name></expr>;</return>
}</block></function>


<function><type><name>void</name></type> <name><name>Solver</name><operator>::</operator><name>toDimacs</name></name><parameter_list>(<parameter><decl><type><name>FILE</name><modifier>*</modifier></type> <name>f</name></decl></parameter>, <parameter><decl><type><name>Clause</name><modifier>&amp;</modifier></type> <name>c</name></decl></parameter>, <parameter><decl><type><name><name>vec</name><argument_list type="generic">&lt;<argument><expr><name>Var</name></expr></argument>&gt;</argument_list></name><modifier>&amp;</modifier></type> <name>map</name></decl></parameter>, <parameter><decl><type><name>Var</name><modifier>&amp;</modifier></type> <name>max</name></decl></parameter>)</parameter_list>
<block>{
    <if>if <condition>(<expr><call><name>satisfied</name><argument_list>(<argument><expr><name>c</name></expr></argument>)</argument_list></call></expr>)</condition><then> <block type="pseudo"><return>return;</return></block></then></if>

    <for>for <control>(<init><decl><type><name>int</name></type> <name>i</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</init> <condition><expr><name>i</name> <operator>&lt;</operator> <call><name><name>c</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>i</name><operator>++</operator></expr></incr>)</control>
        <block type="pseudo"><if>if <condition>(<expr><call><name>value</name><argument_list>(<argument><expr><name><name>c</name><index>[<expr><name>i</name></expr>]</index></name></expr></argument>)</argument_list></call> <operator>!=</operator> <name>l_False</name></expr>)</condition><then>
            <block type="pseudo"><expr_stmt><expr><call><name>fprintf</name><argument_list>(<argument><expr><name>f</name></expr></argument>, <argument><expr><literal type="string">"%s%d "</literal></expr></argument>, <argument><expr><ternary><condition><expr><call><name>sign</name><argument_list>(<argument><expr><name><name>c</name><index>[<expr><name>i</name></expr>]</index></name></expr></argument>)</argument_list></call></expr> ?</condition><then> <expr><literal type="string">"-"</literal></expr> </then><else>: <expr><literal type="string">""</literal></expr></else></ternary></expr></argument>, <argument><expr><call><name>mapVar</name><argument_list>(<argument><expr><call><name>var</name><argument_list>(<argument><expr><name><name>c</name><index>[<expr><name>i</name></expr>]</index></name></expr></argument>)</argument_list></call></expr></argument>, <argument><expr><name>map</name></expr></argument>, <argument><expr><name>max</name></expr></argument>)</argument_list></call><operator>+</operator><literal type="number">1</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt></block></then></if></block></for>
    <expr_stmt><expr><call><name>fprintf</name><argument_list>(<argument><expr><name>f</name></expr></argument>, <argument><expr><literal type="string">"0\n"</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt>
}</block></function>


<function><type><name>void</name></type> <name><name>Solver</name><operator>::</operator><name>toDimacs</name></name><parameter_list>(<parameter><decl><type><specifier>const</specifier> <name>char</name> <modifier>*</modifier></type><name>file</name></decl></parameter>, <parameter><decl><type><specifier>const</specifier> <name><name>vec</name><argument_list type="generic">&lt;<argument><expr><name>Lit</name></expr></argument>&gt;</argument_list></name><modifier>&amp;</modifier></type> <name>assumps</name></decl></parameter>)</parameter_list>
<block>{
    <decl_stmt><decl><type><name>FILE</name><modifier>*</modifier></type> <name>f</name> <init>= <expr><call><name>fopen</name><argument_list>(<argument><expr><name>file</name></expr></argument>, <argument><expr><literal type="string">"wr"</literal></expr></argument>)</argument_list></call></expr></init></decl>;</decl_stmt>
    <if>if <condition>(<expr><name>f</name> <operator>==</operator> <name>NULL</name></expr>)</condition><then>
        <block type="pseudo"><expr_stmt><expr><call><name>fprintf</name><argument_list>(<argument><expr><name>stderr</name></expr></argument>, <argument><expr><literal type="string">"could not open file %s\n"</literal></expr></argument>, <argument><expr><name>file</name></expr></argument>)</argument_list></call></expr><operator>,</operator> <expr><call><name>exit</name><argument_list>(<argument><expr><literal type="number">1</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt></block></then></if>
    <expr_stmt><expr><call><name>toDimacs</name><argument_list>(<argument><expr><name>f</name></expr></argument>, <argument><expr><name>assumps</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <expr_stmt><expr><call><name>fclose</name><argument_list>(<argument><expr><name>f</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
}</block></function>


<function><type><name>void</name></type> <name><name>Solver</name><operator>::</operator><name>toDimacs</name></name><parameter_list>(<parameter><decl><type><name>FILE</name><modifier>*</modifier></type> <name>f</name></decl></parameter>, <parameter><decl><type><specifier>const</specifier> <name><name>vec</name><argument_list type="generic">&lt;<argument><expr><name>Lit</name></expr></argument>&gt;</argument_list></name><modifier>&amp;</modifier></type> <name>assumps</name></decl></parameter>)</parameter_list>
<block>{
    <comment type="line">// Handle case when solver is in contradictory state:</comment>
    <if>if <condition>(<expr><operator>!</operator><name>ok</name></expr>)</condition><then><block>{
        <expr_stmt><expr><call><name>fprintf</name><argument_list>(<argument><expr><name>f</name></expr></argument>, <argument><expr><literal type="string">"p cnf 1 2\n1 0\n-1 0\n"</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt>
        <return>return;</return> }</block></then></if>

    <decl_stmt><decl><type><name><name>vec</name><argument_list type="generic">&lt;<argument><expr><name>Var</name></expr></argument>&gt;</argument_list></name></type> <name>map</name></decl>;</decl_stmt> <decl_stmt><decl><type><name>Var</name></type> <name>max</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</decl_stmt>

    <comment type="line">// Cannot use removeClauses here because it is not safe</comment>
    <comment type="line">// to deallocate them at this point. Could be improved.</comment>
    <decl_stmt><decl><type><name>int</name></type> <name>cnt</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</decl_stmt>
    <for>for <control>(<init><decl><type><name>int</name></type> <name>i</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</init> <condition><expr><name>i</name> <operator>&lt;</operator> <call><name><name>clauses</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>i</name><operator>++</operator></expr></incr>)</control>
        <block type="pseudo"><if>if <condition>(<expr><operator>!</operator><call><name>satisfied</name><argument_list>(<argument><expr><name><name>ca</name><index>[<expr><name><name>clauses</name><index>[<expr><name>i</name></expr>]</index></name></expr>]</index></name></expr></argument>)</argument_list></call></expr>)</condition><then>
            <block type="pseudo"><expr_stmt><expr><name>cnt</name><operator>++</operator></expr>;</expr_stmt></block></then></if></block></for>
        
    <for>for <control>(<init><decl><type><name>int</name></type> <name>i</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</init> <condition><expr><name>i</name> <operator>&lt;</operator> <call><name><name>clauses</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>i</name><operator>++</operator></expr></incr>)</control>
        <block type="pseudo"><if>if <condition>(<expr><operator>!</operator><call><name>satisfied</name><argument_list>(<argument><expr><name><name>ca</name><index>[<expr><name><name>clauses</name><index>[<expr><name>i</name></expr>]</index></name></expr>]</index></name></expr></argument>)</argument_list></call></expr>)</condition><then><block>{
            <decl_stmt><decl><type><name>Clause</name><modifier>&amp;</modifier></type> <name>c</name> <init>= <expr><name><name>ca</name><index>[<expr><name><name>clauses</name><index>[<expr><name>i</name></expr>]</index></name></expr>]</index></name></expr></init></decl>;</decl_stmt>
            <for>for <control>(<init><decl><type><name>int</name></type> <name>j</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</init> <condition><expr><name>j</name> <operator>&lt;</operator> <call><name><name>c</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>j</name><operator>++</operator></expr></incr>)</control>
                <block type="pseudo"><if>if <condition>(<expr><call><name>value</name><argument_list>(<argument><expr><name><name>c</name><index>[<expr><name>j</name></expr>]</index></name></expr></argument>)</argument_list></call> <operator>!=</operator> <name>l_False</name></expr>)</condition><then>
                    <block type="pseudo"><expr_stmt><expr><call><name>mapVar</name><argument_list>(<argument><expr><call><name>var</name><argument_list>(<argument><expr><name><name>c</name><index>[<expr><name>j</name></expr>]</index></name></expr></argument>)</argument_list></call></expr></argument>, <argument><expr><name>map</name></expr></argument>, <argument><expr><name>max</name></expr></argument>)</argument_list></call></expr>;</expr_stmt></block></then></if></block></for>
        }</block></then></if></block></for>

    <comment type="line">// Assumptions are added as unit clauses:</comment>
    <expr_stmt><expr><name>cnt</name> <operator>+=</operator> <call><name><name>assumptions</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</expr_stmt>

    <expr_stmt><expr><call><name>fprintf</name><argument_list>(<argument><expr><name>f</name></expr></argument>, <argument><expr><literal type="string">"p cnf %d %d\n"</literal></expr></argument>, <argument><expr><name>max</name></expr></argument>, <argument><expr><name>cnt</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>

    <for>for <control>(<init><decl><type><name>int</name></type> <name>i</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</init> <condition><expr><name>i</name> <operator>&lt;</operator> <call><name><name>assumptions</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>i</name><operator>++</operator></expr></incr>)</control><block>{
        <expr_stmt><expr><call><name>assert</name><argument_list>(<argument><expr><call><name>value</name><argument_list>(<argument><expr><name><name>assumptions</name><index>[<expr><name>i</name></expr>]</index></name></expr></argument>)</argument_list></call> <operator>!=</operator> <name>l_False</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
        <expr_stmt><expr><call><name>fprintf</name><argument_list>(<argument><expr><name>f</name></expr></argument>, <argument><expr><literal type="string">"%s%d 0\n"</literal></expr></argument>, <argument><expr><ternary><condition><expr><call><name>sign</name><argument_list>(<argument><expr><name><name>assumptions</name><index>[<expr><name>i</name></expr>]</index></name></expr></argument>)</argument_list></call></expr> ?</condition><then> <expr><literal type="string">"-"</literal></expr> </then><else>: <expr><literal type="string">""</literal></expr></else></ternary></expr></argument>, <argument><expr><call><name>mapVar</name><argument_list>(<argument><expr><call><name>var</name><argument_list>(<argument><expr><name><name>assumptions</name><index>[<expr><name>i</name></expr>]</index></name></expr></argument>)</argument_list></call></expr></argument>, <argument><expr><name>map</name></expr></argument>, <argument><expr><name>max</name></expr></argument>)</argument_list></call><operator>+</operator><literal type="number">1</literal></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    }</block></for>

    <for>for <control>(<init><decl><type><name>int</name></type> <name>i</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</init> <condition><expr><name>i</name> <operator>&lt;</operator> <call><name><name>clauses</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>i</name><operator>++</operator></expr></incr>)</control>
        <block type="pseudo"><expr_stmt><expr><call><name>toDimacs</name><argument_list>(<argument><expr><name>f</name></expr></argument>, <argument><expr><name><name>ca</name><index>[<expr><name><name>clauses</name><index>[<expr><name>i</name></expr>]</index></name></expr>]</index></name></expr></argument>, <argument><expr><name>map</name></expr></argument>, <argument><expr><name>max</name></expr></argument>)</argument_list></call></expr>;</expr_stmt></block></for>

    <if>if <condition>(<expr><name>verbosity</name> <operator>&gt;</operator> <literal type="number">0</literal></expr>)</condition><then>
        <block type="pseudo"><expr_stmt><expr><call><name>printf</name><argument_list>(<argument><expr><literal type="string">"Wrote %d clauses with %d variables.\n"</literal></expr></argument>, <argument><expr><name>cnt</name></expr></argument>, <argument><expr><name>max</name></expr></argument>)</argument_list></call></expr>;</expr_stmt></block></then></if>
}</block></function>


<comment type="line">//=================================================================================================</comment>
<comment type="line">// Garbage Collection methods:</comment>

<function><type><name>void</name></type> <name><name>Solver</name><operator>::</operator><name>relocAll</name></name><parameter_list>(<parameter><decl><type><name>ClauseAllocator</name><modifier>&amp;</modifier></type> <name>to</name></decl></parameter>)</parameter_list>
<block>{
    <comment type="line">// All watchers:</comment>
    <comment type="line">//</comment>
    <comment type="line">// for (int i = 0; i &lt; watches.size(); i++)</comment>
    <expr_stmt><expr><call><name><name>watches</name><operator>.</operator><name>cleanAll</name></name><argument_list>()</argument_list></call></expr>;</expr_stmt>
    <for>for <control>(<init><decl><type><name>int</name></type> <name>v</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</init> <condition><expr><name>v</name> <operator>&lt;</operator> <call><name>nVars</name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>v</name><operator>++</operator></expr></incr>)</control>
        <block type="pseudo"><for>for <control>(<init><decl><type><name>int</name></type> <name>s</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</init> <condition><expr><name>s</name> <operator>&lt;</operator> <literal type="number">2</literal></expr>;</condition> <incr><expr><name>s</name><operator>++</operator></expr></incr>)</control><block>{
            <decl_stmt><decl><type><name>Lit</name></type> <name>p</name> <init>= <expr><call><name>mkLit</name><argument_list>(<argument><expr><name>v</name></expr></argument>, <argument><expr><name>s</name></expr></argument>)</argument_list></call></expr></init></decl>;</decl_stmt>
            <comment type="line">// printf(" &gt;&gt;&gt; RELOCING: %s%d\n", sign(p)?"-":"", var(p)+1);</comment>
            <decl_stmt><decl><type><name><name>vec</name><argument_list type="generic">&lt;<argument><expr><name>Watcher</name></expr></argument>&gt;</argument_list></name><modifier>&amp;</modifier></type> <name>ws</name> <init>= <expr><name><name>watches</name><index>[<expr><name>p</name></expr>]</index></name></expr></init></decl>;</decl_stmt>
            <for>for <control>(<init><decl><type><name>int</name></type> <name>j</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</init> <condition><expr><name>j</name> <operator>&lt;</operator> <call><name><name>ws</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>j</name><operator>++</operator></expr></incr>)</control>
                <block type="pseudo"><expr_stmt><expr><call><name><name>ca</name><operator>.</operator><name>reloc</name></name><argument_list>(<argument><expr><name><name>ws</name><index>[<expr><name>j</name></expr>]</index></name><operator>.</operator><name>cref</name></expr></argument>, <argument><expr><name>to</name></expr></argument>)</argument_list></call></expr>;</expr_stmt></block></for>
        }</block></for></block></for>

    <comment type="line">// All reasons:</comment>
    <comment type="line">//</comment>
    <for>for <control>(<init><decl><type><name>int</name></type> <name>i</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</init> <condition><expr><name>i</name> <operator>&lt;</operator> <call><name><name>trail</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>i</name><operator>++</operator></expr></incr>)</control><block>{
        <decl_stmt><decl><type><name>Var</name></type> <name>v</name> <init>= <expr><call><name>var</name><argument_list>(<argument><expr><name><name>trail</name><index>[<expr><name>i</name></expr>]</index></name></expr></argument>)</argument_list></call></expr></init></decl>;</decl_stmt>

        <if>if <condition>(<expr><call><name>reason</name><argument_list>(<argument><expr><name>v</name></expr></argument>)</argument_list></call> <operator>!=</operator> <name>CRef_Undef</name> <operator>&amp;&amp;</operator> <operator>(</operator><name><name>ca</name><index>[<expr><call><name>reason</name><argument_list>(<argument><expr><name>v</name></expr></argument>)</argument_list></call></expr>]</index></name><operator>.</operator><call><name>reloced</name><argument_list>()</argument_list></call> <operator>||</operator> <call><name>locked</name><argument_list>(<argument><expr><name><name>ca</name><index>[<expr><call><name>reason</name><argument_list>(<argument><expr><name>v</name></expr></argument>)</argument_list></call></expr>]</index></name></expr></argument>)</argument_list></call><operator>)</operator></expr>)</condition><then>
            <block type="pseudo"><expr_stmt><expr><call><name><name>ca</name><operator>.</operator><name>reloc</name></name><argument_list>(<argument><expr><name><name>vardata</name><index>[<expr><name>v</name></expr>]</index></name><operator>.</operator><name>reason</name></expr></argument>, <argument><expr><name>to</name></expr></argument>)</argument_list></call></expr>;</expr_stmt></block></then></if>
    }</block></for>

    <comment type="line">// All learnt:</comment>
    <comment type="line">//</comment>
    <for>for <control>(<init><decl><type><name>int</name></type> <name>i</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</init> <condition><expr><name>i</name> <operator>&lt;</operator> <call><name><name>learnts</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>i</name><operator>++</operator></expr></incr>)</control>
        <block type="pseudo"><expr_stmt><expr><call><name><name>ca</name><operator>.</operator><name>reloc</name></name><argument_list>(<argument><expr><name><name>learnts</name><index>[<expr><name>i</name></expr>]</index></name></expr></argument>, <argument><expr><name>to</name></expr></argument>)</argument_list></call></expr>;</expr_stmt></block></for>

    <comment type="line">// All original:</comment>
    <comment type="line">//</comment>
    <for>for <control>(<init><decl><type><name>int</name></type> <name>i</name> <init>= <expr><literal type="number">0</literal></expr></init></decl>;</init> <condition><expr><name>i</name> <operator>&lt;</operator> <call><name><name>clauses</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call></expr>;</condition> <incr><expr><name>i</name><operator>++</operator></expr></incr>)</control>
        <block type="pseudo"><expr_stmt><expr><call><name><name>ca</name><operator>.</operator><name>reloc</name></name><argument_list>(<argument><expr><name><name>clauses</name><index>[<expr><name>i</name></expr>]</index></name></expr></argument>, <argument><expr><name>to</name></expr></argument>)</argument_list></call></expr>;</expr_stmt></block></for>
}</block></function>


<function><type><name>void</name></type> <name><name>Solver</name><operator>::</operator><name>garbageCollect</name></name><parameter_list>()</parameter_list>
<block>{
    <comment type="line">// Initialize the next region to a size corresponding to the estimated utilization degree. This</comment>
    <comment type="line">// is not precise but should avoid some unnecessary reallocations for the new region:</comment>
    <decl_stmt><decl><type><name>ClauseAllocator</name></type> <name>to</name><argument_list>(<argument><expr><call><name><name>ca</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call> <operator>-</operator> <call><name><name>ca</name><operator>.</operator><name>wasted</name></name><argument_list>()</argument_list></call></expr></argument>)</argument_list></decl>;</decl_stmt> 

    <expr_stmt><expr><call><name>relocAll</name><argument_list>(<argument><expr><name>to</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
    <if>if <condition>(<expr><name>verbosity</name> <operator>&gt;=</operator> <literal type="number">2</literal></expr>)</condition><then>
        <block type="pseudo"><expr_stmt><expr><call><name>printf</name><argument_list>(<argument><expr><literal type="string">"|  Garbage collection:   %12d bytes =&gt; %12d bytes             |\n"</literal></expr></argument>, 
               <argument><expr><call><name><name>ca</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call><operator>*</operator><name><name>ClauseAllocator</name><operator>::</operator><name>Unit_Size</name></name></expr></argument>, <argument><expr><call><name><name>to</name><operator>.</operator><name>size</name></name><argument_list>()</argument_list></call><operator>*</operator><name><name>ClauseAllocator</name><operator>::</operator><name>Unit_Size</name></name></expr></argument>)</argument_list></call></expr>;</expr_stmt></block></then></if>
    <expr_stmt><expr><call><name><name>to</name><operator>.</operator><name>moveTo</name></name><argument_list>(<argument><expr><name>ca</name></expr></argument>)</argument_list></call></expr>;</expr_stmt>
}</block></function>
</unit>


and these are the available parents each with his fitness. The lowest fitness the better the parent.
Available parents:
 Parent 1:
 with fitness 11.8873
Parent 1 has 1 edits: ["SrcmlStmtDeletion(('core/Solver.cc.xml', 'stmt', 349))"]
 Parent 2:
 with fitness 11.9353
Parent 2 has 1 edits: ["SrcmlStmtInsertion(('core/Solver.cc.xml', '_inter_block', 386), ('core/Solver.cc.xml', 'stmt', 20))"]
 Parent 3:
 with fitness 11.956
Parent 3 has 1 edits: ["SrcmlStmtInsertion(('core/Solver.cc.xml', '_inter_block', 219), ('core/Solver.cc.xml', 'stmt', 152))"]
 Parent 4:
 with fitness 11.9941
Parent 4 has 1 edits: ["SrcmlStmtInsertion(('core/Solver.cc.xml', '_inter_block', 418), ('core/Solver.cc.xml', 'stmt', 237))"]


Remember you are assisting in the crossover between the parents. Choose as many and whichever edits from the available parents you think will lead to the best child. The proposed edits must be a combination of the available edits. Respond back with the child in the form of a list of edits in the same format as the parents are.
Your response must adhere to the text format: Child: ***the child***. 
