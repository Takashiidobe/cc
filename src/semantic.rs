use std::collections::BTreeMap;

use crate::parse::{
    Const, Decl, DeclKind, Expr, ExprKind, ForInit, FunctionDecl, ParameterDecl, Program, Stmt,
    StmtKind, StorageClass, Type, VariableDecl,
};

#[derive(Clone)]
struct Symbol {
    unique: String,
    ty: Type,
}

#[derive(Clone)]
struct FunctionSignature {
    return_type: Type,
    param_types: Vec<Type>,
}

#[derive(Default)]
pub struct SemanticAnalyzer {
    counter: usize,
    scopes: Vec<BTreeMap<String, Symbol>>,
    functions: BTreeMap<String, FunctionSignature>,
    current_return_type: Option<Type>,
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn analyze_program(mut self, program: Program) -> Program {
        self.enter_scope();

        for decl in &program.0 {
            match &decl.kind {
                DeclKind::Variable(var) => {
                    self.insert_symbol(
                        var.name.clone(),
                        Symbol {
                            unique: var.name.clone(),
                            ty: var.r#type.clone(),
                        },
                    );
                }
                DeclKind::Function(func) => {
                    let signature = FunctionSignature {
                        return_type: func.return_type.clone(),
                        param_types: func.params.iter().map(|p| p.r#type.clone()).collect(),
                    };
                    self.functions.insert(func.name.clone(), signature);
                }
            }
        }

        let decls = program
            .0
            .into_iter()
            .map(|decl| self.analyze_decl(decl))
            .collect();
        self.exit_scope();
        Program(decls)
    }

    fn analyze_decl(&mut self, decl: Decl) -> Decl {
        let Decl {
            kind,
            start,
            end,
            source,
        } = decl;

        let kind = match kind {
            DeclKind::Function(func) => DeclKind::Function(self.analyze_function(func)),
            DeclKind::Variable(var) => DeclKind::Variable(self.analyze_global_variable(var)),
        };

        Decl {
            kind,
            start,
            end,
            source,
        }
    }

    fn analyze_function(&mut self, decl: FunctionDecl) -> FunctionDecl {
        let FunctionDecl {
            name,
            params,
            body,
            storage_class,
            return_type,
        } = decl;

        match body {
            Some(body_stmts) => {
                let prev_return = self.current_return_type.replace(return_type.clone());
                self.enter_scope();

                let mut unique_params = Vec::new();
                for param in params {
                    if param.r#type == Type::Void {
                        panic!("parameter declared with void type");
                    }

                    let unique = self.fresh_name(&param.name);
                    let ty = param.r#type.clone();
                    self.insert_symbol(
                        param.name.clone(),
                        Symbol {
                            unique: unique.clone(),
                            ty: ty.clone(),
                        },
                    );
                    unique_params.push(ParameterDecl {
                        name: unique,
                        r#type: ty,
                    });
                }

                let body = body_stmts
                    .into_iter()
                    .map(|stmt| self.analyze_stmt(stmt))
                    .collect();

                self.exit_scope();
                self.current_return_type = prev_return;

                FunctionDecl {
                    name,
                    params: unique_params,
                    body: Some(body),
                    storage_class,
                    return_type,
                }
            }
            None => FunctionDecl {
                name,
                params,
                body: None,
                storage_class,
                return_type,
            },
        }
    }

    fn analyze_global_variable(&mut self, mut decl: VariableDecl) -> VariableDecl {
        if decl.r#type == Type::Void {
            panic!("variable '{}' declared with void type", decl.name);
        }

        let init = decl.init.take().map(|expr| self.analyze_expr(expr));
        decl.init = init;

        decl
    }

    fn analyze_local_variable_decl(&mut self, mut decl: VariableDecl) -> VariableDecl {
        if decl.r#type == Type::Void {
            panic!("variable '{}' declared with void type", decl.name);
        }

        let init = decl.init.take().map(|expr| {
            let analyzed = self.analyze_expr(expr);
            self.ensure_assignable(&decl.r#type, &analyzed.r#type, "variable initialization");
            analyzed
        });
        decl.init = init;

        match decl.storage_class {
            Some(StorageClass::Extern) => {
                self.insert_symbol(
                    decl.name.clone(),
                    Symbol {
                        unique: decl.name.clone(),
                        ty: decl.r#type.clone(),
                    },
                );
                decl
            }
            Some(StorageClass::Static) => {
                panic!("static local variables are not supported yet");
            }
            None => {
                let original = decl.name.clone();
                let unique = self.fresh_name(&original);
                self.insert_symbol(
                    original,
                    Symbol {
                        unique: unique.clone(),
                        ty: decl.r#type.clone(),
                    },
                );
                decl.name = unique;
                decl
            }
        }
    }

    fn analyze_stmt(&mut self, stmt: Stmt) -> Stmt {
        let Stmt {
            kind,
            start,
            end,
            source,
            r#type: _,
        } = stmt;

        let kind = match kind {
            StmtKind::Return(expr) => {
                let expr = self.analyze_expr(expr);
                if let Some(expected) = &self.current_return_type {
                    self.ensure_assignable(expected, &expr.r#type, "return statement");
                }
                StmtKind::Return(expr)
            }
            StmtKind::Expr(expr) => StmtKind::Expr(self.analyze_expr(expr)),
            StmtKind::Compound(stmts) => {
                self.enter_scope();
                let stmts = stmts.into_iter().map(|s| self.analyze_stmt(s)).collect();
                self.exit_scope();
                StmtKind::Compound(stmts)
            }
            StmtKind::Declaration(decl) => {
                let decl = self.analyze_local_variable_decl(decl);
                StmtKind::Declaration(decl)
            }
            StmtKind::Null => StmtKind::Null,
            StmtKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let condition = self.analyze_expr(condition);
                self.ensure_integer_type(&condition.r#type, "if condition");
                StmtKind::If {
                    condition,
                    then_branch: Box::new(self.analyze_stmt(*then_branch)),
                    else_branch: else_branch.map(|stmt| Box::new(self.analyze_stmt(*stmt))),
                }
            }
            StmtKind::While {
                condition,
                body,
                loop_id,
            } => {
                let condition = self.analyze_expr(condition);
                self.ensure_integer_type(&condition.r#type, "while condition");
                StmtKind::While {
                    condition,
                    body: Box::new(self.analyze_stmt(*body)),
                    loop_id,
                }
            }
            StmtKind::DoWhile {
                body,
                condition,
                loop_id,
            } => {
                let condition = self.analyze_expr(condition);
                self.ensure_integer_type(&condition.r#type, "do-while condition");
                StmtKind::DoWhile {
                    body: Box::new(self.analyze_stmt(*body)),
                    condition,
                    loop_id,
                }
            }
            StmtKind::For {
                init,
                condition,
                post,
                body,
                loop_id,
            } => {
                self.enter_scope();
                let init = self.analyze_for_init(init);
                let condition = condition.map(|expr| {
                    let analyzed = self.analyze_expr(expr);
                    self.ensure_integer_type(&analyzed.r#type, "for condition");
                    analyzed
                });
                let post = post.map(|expr| self.analyze_expr(expr));
                let body = Box::new(self.analyze_stmt(*body));
                self.exit_scope();
                StmtKind::For {
                    init,
                    condition,
                    post,
                    body,
                    loop_id,
                }
            }
            StmtKind::Break { loop_id } => StmtKind::Break { loop_id },
            StmtKind::Continue { loop_id } => StmtKind::Continue { loop_id },
        };

        Stmt {
            kind,
            start,
            end,
            source,
            r#type: Type::Void,
        }
    }

    fn analyze_expr(&mut self, expr: Expr) -> Expr {
        let Expr {
            kind,
            start,
            end,
            source,
            r#type: _,
        } = expr;

        match kind {
            ExprKind::Constant(c) => {
                let ty = match &c {
                    Const::Int(_) => Type::Int,
                    Const::Long(_) => Type::Long,
                };
                Expr {
                    kind: ExprKind::Constant(c),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
            ExprKind::Var(name) => {
                let symbol = self
                    .lookup_symbol(&name)
                    .unwrap_or_else(|| panic!("use of undeclared identifier {name}"));
                Expr {
                    kind: ExprKind::Var(symbol.unique.clone()),
                    start,
                    end,
                    source,
                    r#type: symbol.ty.clone(),
                }
            }
            ExprKind::FunctionCall(name, args) => {
                let signature = self
                    .functions
                    .get(&name)
                    .cloned()
                    .unwrap_or_else(|| panic!("call to undeclared function {name}"));
                if args.len() != signature.param_types.len() {
                    panic!(
                        "function '{}' called with wrong number of arguments (expected {}, got {})",
                        name,
                        signature.param_types.len(),
                        args.len()
                    );
                }

                let analyzed_args = args
                    .into_iter()
                    .zip(signature.param_types.iter())
                    .map(|(arg, expected)| {
                        let expr = self.analyze_expr(arg);
                        self.ensure_assignable(expected, &expr.r#type, "function argument");
                        expr
                    })
                    .collect();

                Expr {
                    kind: ExprKind::FunctionCall(name, analyzed_args),
                    start,
                    end,
                    source,
                    r#type: signature.return_type,
                }
            }
            ExprKind::Cast(target, expr) => {
                if target == Type::Void {
                    panic!("cannot cast to void type");
                }

                let expr = self.analyze_expr(*expr);
                self.ensure_integer_type(&expr.r#type, "cast operand");
                let ty = target.clone();
                Expr {
                    kind: ExprKind::Cast(ty.clone(), Box::new(expr)),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
            ExprKind::Neg(expr) => {
                self.analyze_numeric_unary(*expr, start, end, source, ExprKind::Neg)
            }
            ExprKind::BitNot(expr) => {
                self.analyze_numeric_unary(*expr, start, end, source, ExprKind::BitNot)
            }
            ExprKind::Not(expr) => {
                let expr = self.analyze_expr(*expr);
                self.ensure_integer_type(&expr.r#type, "logical not");
                Expr {
                    kind: ExprKind::Not(Box::new(expr)),
                    start,
                    end,
                    source,
                    r#type: Type::Int,
                }
            }
            ExprKind::Conditional(cond, then_expr, else_expr) => {
                let cond = self.analyze_expr(*cond);
                self.ensure_integer_type(&cond.r#type, "conditional condition");
                let then_expr = self.analyze_expr(*then_expr);
                self.ensure_integer_type(&then_expr.r#type, "conditional arm");
                let else_expr = self.analyze_expr(*else_expr);
                self.ensure_integer_type(&else_expr.r#type, "conditional arm");
                let ty = self.numeric_result_type(&then_expr.r#type, &else_expr.r#type);
                Expr {
                    kind: ExprKind::Conditional(
                        Box::new(cond),
                        Box::new(then_expr),
                        Box::new(else_expr),
                    ),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
            ExprKind::Add(lhs, rhs) => {
                self.analyze_numeric_binary(*lhs, *rhs, start, end, source, ExprKind::Add)
            }
            ExprKind::Sub(lhs, rhs) => {
                self.analyze_numeric_binary(*lhs, *rhs, start, end, source, ExprKind::Sub)
            }
            ExprKind::Mul(lhs, rhs) => {
                self.analyze_numeric_binary(*lhs, *rhs, start, end, source, ExprKind::Mul)
            }
            ExprKind::Div(lhs, rhs) => {
                self.analyze_numeric_binary(*lhs, *rhs, start, end, source, ExprKind::Div)
            }
            ExprKind::Rem(lhs, rhs) => {
                self.analyze_numeric_binary(*lhs, *rhs, start, end, source, ExprKind::Rem)
            }
            ExprKind::Equal(lhs, rhs) => {
                self.analyze_comparison(*lhs, *rhs, start, end, source, ExprKind::Equal)
            }
            ExprKind::NotEqual(lhs, rhs) => {
                self.analyze_comparison(*lhs, *rhs, start, end, source, ExprKind::NotEqual)
            }
            ExprKind::LessThan(lhs, rhs) => {
                self.analyze_comparison(*lhs, *rhs, start, end, source, ExprKind::LessThan)
            }
            ExprKind::LessThanEqual(lhs, rhs) => {
                self.analyze_comparison(*lhs, *rhs, start, end, source, ExprKind::LessThanEqual)
            }
            ExprKind::GreaterThan(lhs, rhs) => {
                self.analyze_comparison(*lhs, *rhs, start, end, source, ExprKind::GreaterThan)
            }
            ExprKind::GreaterThanEqual(lhs, rhs) => {
                self.analyze_comparison(*lhs, *rhs, start, end, source, ExprKind::GreaterThanEqual)
            }
            ExprKind::Or(lhs, rhs) => {
                self.analyze_logical_binary(*lhs, *rhs, start, end, source, ExprKind::Or)
            }
            ExprKind::And(lhs, rhs) => {
                self.analyze_logical_binary(*lhs, *rhs, start, end, source, ExprKind::And)
            }
            ExprKind::BitAnd(lhs, rhs) => {
                self.analyze_numeric_binary(*lhs, *rhs, start, end, source, ExprKind::BitAnd)
            }
            ExprKind::Xor(lhs, rhs) => {
                self.analyze_numeric_binary(*lhs, *rhs, start, end, source, ExprKind::Xor)
            }
            ExprKind::BitOr(lhs, rhs) => {
                self.analyze_numeric_binary(*lhs, *rhs, start, end, source, ExprKind::BitOr)
            }
            ExprKind::LeftShift(lhs, rhs) => {
                let lhs = self.analyze_expr(*lhs);
                let rhs = self.analyze_expr(*rhs);
                self.ensure_integer_type(&lhs.r#type, "shift operand");
                self.ensure_integer_type(&rhs.r#type, "shift amount");
                let ty = lhs.r#type.clone();
                Expr {
                    kind: ExprKind::LeftShift(Box::new(lhs), Box::new(rhs)),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
            ExprKind::RightShift(lhs, rhs) => {
                let lhs = self.analyze_expr(*lhs);
                let rhs = self.analyze_expr(*rhs);
                self.ensure_integer_type(&lhs.r#type, "shift operand");
                self.ensure_integer_type(&rhs.r#type, "shift amount");
                let ty = lhs.r#type.clone();
                Expr {
                    kind: ExprKind::RightShift(Box::new(lhs), Box::new(rhs)),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
            ExprKind::Assignment(lhs, rhs) => {
                let lhs = self.analyze_expr(*lhs);
                if !matches!(lhs.kind, ExprKind::Var(_)) {
                    panic!("left-hand side of assignment must be a variable");
                }
                let lhs_type = lhs.r#type.clone();
                let rhs = self.analyze_expr(*rhs);
                self.ensure_assignable(&lhs_type, &rhs.r#type, "assignment");
                Expr {
                    kind: ExprKind::Assignment(Box::new(lhs), Box::new(rhs)),
                    start,
                    end,
                    source,
                    r#type: lhs_type,
                }
            }
            ExprKind::PreIncrement(expr) => {
                let expr = self.analyze_expr(*expr);
                if !matches!(expr.kind, ExprKind::Var(_)) {
                    panic!("increment target must be a variable");
                }
                self.ensure_integer_type(&expr.r#type, "pre-increment operand");
                let ty = expr.r#type.clone();
                Expr {
                    kind: ExprKind::PreIncrement(Box::new(expr)),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
            ExprKind::PreDecrement(expr) => {
                let expr = self.analyze_expr(*expr);
                if !matches!(expr.kind, ExprKind::Var(_)) {
                    panic!("decrement target must be a variable");
                }
                self.ensure_integer_type(&expr.r#type, "pre-decrement operand");
                let ty = expr.r#type.clone();
                Expr {
                    kind: ExprKind::PreDecrement(Box::new(expr)),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
            ExprKind::PostIncrement(expr) => {
                let expr = self.analyze_expr(*expr);
                if !matches!(expr.kind, ExprKind::Var(_)) {
                    panic!("increment target must be a variable");
                }
                self.ensure_integer_type(&expr.r#type, "post-increment operand");
                let ty = expr.r#type.clone();
                Expr {
                    kind: ExprKind::PostIncrement(Box::new(expr)),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
            ExprKind::PostDecrement(expr) => {
                let expr = self.analyze_expr(*expr);
                if !matches!(expr.kind, ExprKind::Var(_)) {
                    panic!("decrement target must be a variable");
                }
                self.ensure_integer_type(&expr.r#type, "post-decrement operand");
                let ty = expr.r#type.clone();
                Expr {
                    kind: ExprKind::PostDecrement(Box::new(expr)),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
        }
    }

    fn analyze_for_init(&mut self, init: ForInit) -> ForInit {
        match init {
            ForInit::Declaration(stmt) => ForInit::Declaration(Box::new(self.analyze_stmt(*stmt))),
            ForInit::Expr(expr) => ForInit::Expr(expr.map(|e| self.analyze_expr(e))),
        }
    }

    fn analyze_numeric_unary<F>(
        &mut self,
        expr: Expr,
        start: usize,
        end: usize,
        source: String,
        constructor: F,
    ) -> Expr
    where
        F: FnOnce(Box<Expr>) -> ExprKind,
    {
        let expr = self.analyze_expr(expr);
        self.ensure_integer_type(&expr.r#type, "unary operator");
        let ty = expr.r#type.clone();
        Expr {
            kind: constructor(Box::new(expr)),
            start,
            end,
            source,
            r#type: ty,
        }
    }

    fn analyze_numeric_binary<F>(
        &mut self,
        lhs: Expr,
        rhs: Expr,
        start: usize,
        end: usize,
        source: String,
        constructor: F,
    ) -> Expr
    where
        F: FnOnce(Box<Expr>, Box<Expr>) -> ExprKind,
    {
        let lhs = self.analyze_expr(lhs);
        let rhs = self.analyze_expr(rhs);
        self.ensure_integer_type(&lhs.r#type, "binary operand");
        self.ensure_integer_type(&rhs.r#type, "binary operand");
        let ty = self.numeric_result_type(&lhs.r#type, &rhs.r#type);
        Expr {
            kind: constructor(Box::new(lhs), Box::new(rhs)),
            start,
            end,
            source,
            r#type: ty,
        }
    }

    fn analyze_comparison<F>(
        &mut self,
        lhs: Expr,
        rhs: Expr,
        start: usize,
        end: usize,
        source: String,
        constructor: F,
    ) -> Expr
    where
        F: FnOnce(Box<Expr>, Box<Expr>) -> ExprKind,
    {
        let lhs = self.analyze_expr(lhs);
        let rhs = self.analyze_expr(rhs);
        self.ensure_integer_type(&lhs.r#type, "comparison operand");
        self.ensure_integer_type(&rhs.r#type, "comparison operand");
        Expr {
            kind: constructor(Box::new(lhs), Box::new(rhs)),
            start,
            end,
            source,
            r#type: Type::Int,
        }
    }

    fn analyze_logical_binary<F>(
        &mut self,
        lhs: Expr,
        rhs: Expr,
        start: usize,
        end: usize,
        source: String,
        constructor: F,
    ) -> Expr
    where
        F: FnOnce(Box<Expr>, Box<Expr>) -> ExprKind,
    {
        let lhs = self.analyze_expr(lhs);
        let rhs = self.analyze_expr(rhs);
        self.ensure_integer_type(&lhs.r#type, "logical operand");
        self.ensure_integer_type(&rhs.r#type, "logical operand");
        Expr {
            kind: constructor(Box::new(lhs), Box::new(rhs)),
            start,
            end,
            source,
            r#type: Type::Int,
        }
    }

    fn numeric_result_type(&self, lhs: &Type, rhs: &Type) -> Type {
        match (lhs, rhs) {
            (Type::Long, _) | (_, Type::Long) => Type::Long,
            (Type::Int, Type::Int) => Type::Int,
            _ => panic!(
                "unsupported operand types {:?} and {:?} in numeric expression",
                lhs, rhs
            ),
        }
    }

    fn ensure_integer_type(&self, ty: &Type, context: &str) {
        if !matches!(ty, Type::Int | Type::Long) {
            panic!("{context} requires integer type, found {:?}", ty);
        }
    }

    fn ensure_assignable(&self, target: &Type, value: &Type, context: &str) {
        match (target, value) {
            (Type::Int, Type::Int | Type::Long) => {}
            (Type::Long, Type::Int | Type::Long) => {}
            _ => panic!(
                "{context} requires compatible integer types ({:?} <- {:?})",
                target, value
            ),
        }
    }

    fn insert_symbol(&mut self, name: String, symbol: Symbol) {
        self.scopes
            .last_mut()
            .expect("scope stack empty")
            .insert(name, symbol);
    }

    fn lookup_symbol(&self, name: &str) -> Option<&Symbol> {
        for scope in self.scopes.iter().rev() {
            if let Some(symbol) = scope.get(name) {
                return Some(symbol);
            }
        }
        None
    }

    fn enter_scope(&mut self) {
        self.scopes.push(BTreeMap::new());
    }

    fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    fn fresh_name(&mut self, base: &str) -> String {
        let name = format!("{base}{}", self.counter);
        self.counter += 1;
        name
    }
}
