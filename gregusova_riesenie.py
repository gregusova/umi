# ziak_riesenie.py
"""
Riešenia úloh z linear_algebra.ipynb

Meno žiaka: Nikita Gregusova
Dátum: 14.1.2026
"""

import time
import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import jax.numpy as jnp
import numpy as np
import torch

P = ParamSpec("P")
T = TypeVar("T")

# ============================================================================
# HELPER FUNCTIONS: Time measurement (see example below)
# ============================================================================

def timecheck(func: Callable[P, T]) -> Callable[P, tuple[T, float]]:
    """Decorator to measure function execution time.

    Usage:
        @timecheck
        def my_function():
            return result

        result, elapsed = my_function()
    """

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[T, float]:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        return result, elapsed

    return wrapper


# ============================================================================
# ČASŤ 1: ZÁKLADY
# ============================================================================

def _manual_tensor_times2_plus1(tensor: jnp.ndarray) -> jnp.ndarray:
    """Manuálna implementácia `tensor * 2 + 1` pomocou for-loop.

    Funguje univerzálne aj pre iný tvar (vektor/matica/tenzor) tým, že spracuje
    sploštené dáta a potom vráti pôvodný shape.
    """

    flat = jnp.ravel(tensor)
    result_manual = []
    for i in range(len(flat)):
        result_manual.append(flat[i] * 2 + 1)
    return jnp.array(result_manual).reshape(tensor.shape)


uloha_1_1_su_rovnake: bool | None = None


def uloha_1_1() -> jnp.ndarray:
    """Vytvorí JAX tenzor z Python zoznamu a implementuje operáciu "ručne" pomocou for loop.

    Returns:
        jnp.ndarray: Vektorizovaný výsledok operácie tensor * 2 + 1
    """

    data = [1, 2, 3, 4, 5]
    tensor = jnp.array(data)

    # Manuálne (for-loop) – ZÁKAZ: nepoužiť priamo `tensor * 2 + 1`
    result_manual = _manual_tensor_times2_plus1(tensor)

    # Vektorizované (na porovnanie)
    result_vectorized = tensor * 2 + 1

    # Porovnanie (bez printov pri importe)
    global uloha_1_1_su_rovnake
    uloha_1_1_su_rovnake = bool(jnp.array_equal(result_manual, result_vectorized))

    return result_vectorized


uloha_1_1_vysvetlenie: str = (
    "`* 2` vynásobí každý prvok tenzora číslom 2; `+ 1` pripočíta 1 ku každému prvku. "
    "Vektorizovaná verzia je rýchlejšia, lebo používa optimalizované operácie nad celým tenzorom "
    "(SIMD/BLAS/XLA a menej Python overheadu), namiesto pomalého Python for-loopu." 
)



uloha_1_1_ocakavany_format: str = "jnp.ndarray, shape (5,), integer dtype"


# ---------------------------------------------------------------------------
# Úloha 1.2 – Transpozícia s analýzou a opravou chýb
# ---------------------------------------------------------------------------

uloha_1_2_pokus1_shape: tuple[int, ...] | None = None
uloha_1_2_pokus2_shape: tuple[int, ...] | None = None
uloha_1_2_pokus3_shape: tuple[int, ...] | None = None

uloha_1_2_pokus3_vysvetlenie: str = (
    "Pri 1D vektore (shape (n,)) transpozícia nič nezmení, lebo nie je čo prehodiť: "
    "vektor nemá os 'riadky' a 'stĺpce'. Transpozícia má význam až pre 2D maticu "
    "(napr. (n,1) alebo (1,n)). Preto treba najprv zmeniť shape (reshape) a až potom transponovať."
)


def uloha_1_2() -> jnp.ndarray:
    """Analyzuje transpozíciu a opraví chyby.

    Pokus 1:
        A1 shape (3,2), A1.T -> (2,3)
    Pokus 2:
        jnp.transpose(A2, (1,0)) -> (2,3) (rovnaké ako .T pri 2D)
    Pokus 3:
        A3 shape (6,), A3.T -> (6,) (nezmení sa)

    Returns:
        jnp.ndarray: Transponovaná matica (3, 2) vytvorená z vektora (6,)
    """

    # Pokus 1
    A1 = jnp.array([[1, 2], [3, 4], [5, 6]])  # shape (3, 2)
    result1 = A1.T

    # Pokus 2
    A2 = jnp.array([[1, 2], [3, 4], [5, 6]])
    result2 = jnp.transpose(A2, (1, 0))

    # Pokus 3
    A3 = jnp.array([1, 2, 3, 4, 5, 6])  # shape (6,)
    result3 = A3.T

    # Uloženie shape analýzy (bez printovania pri importe)
    global uloha_1_2_pokus1_shape, uloha_1_2_pokus2_shape, uloha_1_2_pokus3_shape
    uloha_1_2_pokus1_shape = tuple(result1.shape)
    uloha_1_2_pokus2_shape = tuple(result2.shape)
    uloha_1_2_pokus3_shape = tuple(result3.shape)

    # Oprava pokusu 3:
    # z vektora (6,) urobíme maticu (2,3) a transponujeme na (3,2)
    matica_2_3 = A3.reshape((2, 3))
    transponovana_matica = matica_2_3.T
    return transponovana_matica


# ---------------------------------------------------------------------------
# Úloha 1.3 – Broadcasting (ručná implementácia + porovnanie)
# ---------------------------------------------------------------------------

uloha_1_3_su_rovnake: bool | None = None
uloha_1_3_cas_manual: float | None = None
uloha_1_3_cas_vectorized: float | None = None
uloha_1_3_result3_error: str | None = None

uloha_1_3_analyza: str = (
    "result1: A * v je správne, lebo broadcasting rozšíri (4,) na (1,4) po stĺpcoch. "
    "result2: A * v.reshape(1,4) je ekvivalentné a tiež správne. "
    "result3: A * v.reshape(4,1) je nesprávne, lebo (3,4) a (4,1) sa nedajú broadcastovať "
    "(nesedí prvý rozmer 3 vs 4)."
)


def uloha_1_3(A: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Implementuje broadcasting "ručne" pomocou for loop a porovná s vektorizovanou verziou.

    Args:
        A: Matica (3, 4) (všeobecne (m, n))
        v: Vektor (4,) (všeobecne (n,))

    Returns:
        jnp.ndarray: Výsledok broadcasting A * v (každý stĺpec A vynásobený v[j])
    """

    if A.ndim != 2:
        raise ValueError("A musí byť 2D matica")
    if v.ndim != 1:
        raise ValueError("v musí byť 1D vektor")
    if A.shape[1] != v.shape[0]:
        raise ValueError("Počet stĺpcov A musí sedieť s dĺžkou v")

    m, n = A.shape

    # Ručná implementácia – ZÁKAZ: nepoužiť priamo A * v
    start = time.perf_counter()
    result_manual = []
    for i in range(m):
        row = []
        for j in range(n):
            row.append(A[i, j] * v[j])
        result_manual.append(row)
    result_manual = jnp.array(result_manual)
    # JAX je lazy/asynchrónny – pre férový timing počkáme na výpočet
    result_manual.block_until_ready()
    end = time.perf_counter()
    elapsed_manual = end - start

    # Vektorizované riešenia
    start = time.perf_counter()
    result1 = A * v
    result1.block_until_ready()
    end = time.perf_counter()
    elapsed_vectorized = end - start

    result2 = A * v.reshape(1, -1)

    # "Nesprávny" pokus – očakávame chybu broadcastingu
    err_msg: str | None
    try:
        bad = A * v.reshape(-1, 1)
        bad.block_until_ready()
        err_msg = None
    except Exception as e:  # noqa: BLE001
        err_msg = f"{type(e).__name__}: {e}"

    # Porovnanie
    same = bool(jnp.array_equal(result_manual, result1)) and bool(jnp.array_equal(result1, result2))

    global uloha_1_3_su_rovnake, uloha_1_3_cas_manual, uloha_1_3_cas_vectorized, uloha_1_3_result3_error
    uloha_1_3_su_rovnake = same
    uloha_1_3_cas_manual = elapsed_manual
    uloha_1_3_cas_vectorized = elapsed_vectorized
    uloha_1_3_result3_error = err_msg

    # Bonus: násobenie každého RIADKU vektorom w (m,)
    # Správne: A * w.reshape(m,1) alebo w[:, None]
    w = jnp.arange(1, m + 1) * 10  # napr. [10,20,30] pre m=3
    _row_scaled = A * w.reshape(m, 1)

    return result1


# ---------------------------------------------------------------------------
# Úloha 1.4 – PyTorch verzie úloh 1.1–1.3
# ---------------------------------------------------------------------------

uloha_1_4_1_su_rovnake: bool | None = None
uloha_1_4_2_pokus1_shape: tuple[int, ...] | None = None
uloha_1_4_2_pokus2_shape: tuple[int, ...] | None = None
uloha_1_4_2_pokus3_shape: tuple[int, ...] | None = None
uloha_1_4_2_pokus3_vysvetlenie: str = (
    "Pri 1D torch vektore (shape (n,)) transpozícia nič nezmení, lebo nie je čo prehodiť. "
    "Transpozícia dáva zmysel až pre 2D tenzor (napr. (n,1) alebo (1,n))."
)

uloha_1_4_3_su_rovnake: bool | None = None
uloha_1_4_3_cas_manual: float | None = None
uloha_1_4_3_cas_vectorized: float | None = None
uloha_1_4_3_result3_error: str | None = None


def _torch_sync_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def uloha_1_4_1() -> torch.Tensor:
    """Vytvorí PyTorch tensor z Python zoznamu a porovná ručnú vs vektorizovanú verziu.

    Returns:
        torch.Tensor: Vektorizovaný výsledok operácie tensor * 2 + 1
    """

    data = [1, 2, 3, 4, 5]
    tensor = torch.tensor(data)

    # Manuálne (for-loop) – nepoužiť priamo `tensor * 2 + 1` pre ručnú verziu
    manual_list = []
    for i in range(tensor.shape[0]):
        manual_list.append(tensor[i] * 2 + 1)
    result_manual = torch.stack(manual_list)

    # Vektorizované
    result_vectorized = tensor * 2 + 1

    global uloha_1_4_1_su_rovnake
    uloha_1_4_1_su_rovnake = bool(torch.equal(result_manual, result_vectorized))
    return result_vectorized


def uloha_1_4_2() -> torch.Tensor:
    """Analyzuje transpozíciu v PyTorch a opraví pokus s vektorom.

    Returns:
        torch.Tensor: Transponovaná matica (3, 2) vytvorená z vektora (6,)
    """

    A1 = torch.tensor([[1, 2], [3, 4], [5, 6]])  # (3,2)
    result1 = A1.T

    A2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
    result2 = torch.transpose(A2, 0, 1)

    A3 = torch.tensor([1, 2, 3, 4, 5, 6])  # (6,)
    # .T na 1D vektore je identita (nezmení sa). V novších verziách PyTorch to môže hádzať warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        result3 = A3.T

    global uloha_1_4_2_pokus1_shape, uloha_1_4_2_pokus2_shape, uloha_1_4_2_pokus3_shape
    uloha_1_4_2_pokus1_shape = tuple(result1.shape)
    uloha_1_4_2_pokus2_shape = tuple(result2.shape)
    uloha_1_4_2_pokus3_shape = tuple(result3.shape)

    matica_2_3 = A3.reshape(2, 3)
    transponovana_matica = matica_2_3.T
    return transponovana_matica


def uloha_1_4_3(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Implementuje broadcasting v PyTorch "ručne" pomocou for loop.

    Args:
        A: Matica (3, 4) (všeobecne (m, n))
        v: Vektor (4,) (všeobecne (n,))

    Returns:
        torch.Tensor: Výsledok broadcasting A * v
    """

    if A.ndim != 2:
        raise ValueError("A musí byť 2D matica")
    if v.ndim != 1:
        raise ValueError("v musí byť 1D vektor")
    if A.shape[1] != v.shape[0]:
        raise ValueError("Počet stĺpcov A musí sedieť s dĺžkou v")

    m, n = A.shape

    _torch_sync_if_needed()
    start = time.perf_counter()
    result_manual = torch.empty_like(A)
    for i in range(m):
        for j in range(n):
            result_manual[i, j] = A[i, j] * v[j]
    _torch_sync_if_needed()
    elapsed_manual = time.perf_counter() - start

    _torch_sync_if_needed()
    start = time.perf_counter()
    result_vectorized = A * v
    _torch_sync_if_needed()
    elapsed_vectorized = time.perf_counter() - start

    # Alternatíva s reshape (1,n)
    _ = A * v.reshape(1, -1)

    # Nesprávny pokus (n,1)
    err_msg: str | None
    try:
        _bad = A * v.reshape(-1, 1)
        err_msg = None
    except Exception as e:  # noqa: BLE001
        err_msg = f"{type(e).__name__}: {e}"

    global uloha_1_4_3_su_rovnake, uloha_1_4_3_cas_manual, uloha_1_4_3_cas_vectorized, uloha_1_4_3_result3_error
    uloha_1_4_3_su_rovnake = bool(torch.equal(result_manual, result_vectorized))
    uloha_1_4_3_cas_manual = elapsed_manual
    uloha_1_4_3_cas_vectorized = elapsed_vectorized
    uloha_1_4_3_result3_error = err_msg

    # Bonus: násobenie každého RIADKU vektorom w (m,)
    w = torch.arange(1, m + 1, dtype=A.dtype, device=A.device) * 10
    _row_scaled = A * w.reshape(m, 1)

    return result_vectorized


# ---------------------------------------------------------------------------
# Úloha 1.5 – Porovnanie JAX vs PyTorch
# ---------------------------------------------------------------------------


def uloha_1_5() -> str:
    """Porovná výsledky a výkon operácií v JAX vs PyTorch.

    Returns:
        str: Textové vysvetlenie rozdielov medzi JAX a PyTorch
    """

    # 1) Porovnanie shape a hodnôt na malých príkladoch
    A_j = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=jnp.float32)
    v_j = jnp.array([2, 3, 4, 5], dtype=jnp.float32)
    t_j = jnp.array([1, 2, 3, 4, 5], dtype=jnp.float32)
    j_broadcast = (A_j * v_j)
    j_scalar = (t_j * 2 + 1)
    j_transpose = jnp.array([[1, 2], [3, 4], [5, 6]], dtype=jnp.float32).T

    A_t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.float32)
    v_t = torch.tensor([2, 3, 4, 5], dtype=torch.float32)
    t_t = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    t_broadcast = (A_t * v_t)
    t_scalar = (t_t * 2 + 1)
    t_transpose = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32).T

    same_shapes = (
        j_broadcast.shape == tuple(t_broadcast.shape)
        and j_scalar.shape == tuple(t_scalar.shape)
        and j_transpose.shape == tuple(t_transpose.shape)
    )

    same_values = (
        np.allclose(np.array(j_broadcast), t_broadcast.cpu().numpy())
        and np.allclose(np.array(j_scalar), t_scalar.cpu().numpy())
        and np.allclose(np.array(j_transpose), t_transpose.cpu().numpy())
    )

    # 2) Výkon na väčšom príklade (iba vektorizované operácie)
    m, n = 1024, 1024
    A_j_big = jnp.ones((m, n), dtype=jnp.float32)
    v_j_big = jnp.arange(n, dtype=jnp.float32)

    A_t_big = torch.ones((m, n), dtype=torch.float32)
    v_t_big = torch.arange(n, dtype=torch.float32)

    # JAX timing (block_until_ready)
    start = time.perf_counter()
    jb = (A_j_big * v_j_big)
    jb.block_until_ready()
    j_broadcast_time = time.perf_counter() - start

    start = time.perf_counter()
    jt = A_j_big.T
    jt.block_until_ready()
    j_transpose_time = time.perf_counter() - start

    start = time.perf_counter()
    js = (A_j_big * 2.0 + 1.0)
    js.block_until_ready()
    j_scalar_time = time.perf_counter() - start

    # PyTorch timing
    _torch_sync_if_needed()
    start = time.perf_counter()
    tb = (A_t_big * v_t_big)
    _torch_sync_if_needed()
    t_broadcast_time = time.perf_counter() - start

    _torch_sync_if_needed()
    start = time.perf_counter()
    tt = A_t_big.T
    _torch_sync_if_needed()
    t_transpose_time = time.perf_counter() - start

    _torch_sync_if_needed()
    start = time.perf_counter()
    ts = (A_t_big * 2.0 + 1.0)
    _torch_sync_if_needed()
    t_scalar_time = time.perf_counter() - start

    return (
        f"Shapes rovnaké: {same_shapes}. Hodnoty rovnaké: {same_values}. "
        f"\n\nVýkon (vektorizované, približne; CPU/GPU závisí od prostredia):"
        f"\n- Broadcasting: JAX {j_broadcast_time:.6f}s vs PyTorch {t_broadcast_time:.6f}s"
        f"\n- Transpozícia: JAX {j_transpose_time:.6f}s vs PyTorch {t_transpose_time:.6f}s"
        f"\n- Skalárne operácie: JAX {j_scalar_time:.6f}s vs PyTorch {t_scalar_time:.6f}s"
        "\n\nRozdiely: JAX je funkcionálny a často používa XLA (optimalizácie/JIT), "
        "kdežto PyTorch je eager a dynamický (autograd graf vzniká za behu). "
        "Broadcasting je konceptuálne rovnaký, ale defaultné dtype sa môže líšiť "
        "(napr. JAX často int32, PyTorch často int64 pri int vstupoch). "
        "Pri malých tenzoroch dominuje overhead; pri veľkých zvyčajne vyhráva vektorizácia/optimalizácie."
    )


# ---------------------------------------------------------------------------
# Úloha 1.4 – PyTorch verzie (1.1–1.3)
# ---------------------------------------------------------------------------

uloha_1_4_poznamka_tensor_vs_Tensor: str = (
    "V PyTorch `torch.tensor(data)` typicky inferuje dtype z dát (napr. int -> int64), "
    "kým `torch.Tensor(data)` používa default floating dtype (často float32)."
)


uloha_1_4_1_su_rovnake: bool | None = None


def uloha_1_4_1() -> torch.Tensor:
    """Vytvorí PyTorch tensor z Python zoznamu a porovná ručnú vs. vektorizovanú verziu.

    Returns:
        torch.Tensor: Vektorizovaný výsledok operácie tensor * 2 + 1
    """

    data = [1, 2, 3, 4, 5]
    tensor = torch.tensor(data)

    # Ručne cez for-loop (nepoužiť priamo tensor * 2 + 1 ako jeden výraz)
    result_manual = []
    for i in range(len(tensor)):
        result_manual.append(tensor[i] * 2 + 1)
    result_manual_t = torch.stack(result_manual)

    # Vektorizovane
    result_vectorized = tensor * 2 + 1

    global uloha_1_4_1_su_rovnake
    uloha_1_4_1_su_rovnake = bool(torch.equal(result_manual_t, result_vectorized))
    return result_vectorized


uloha_1_4_2_pokus1_shape: tuple[int, ...] | None = None
uloha_1_4_2_pokus2_shape: tuple[int, ...] | None = None
uloha_1_4_2_pokus3_shape: tuple[int, ...] | None = None

uloha_1_4_2_pokus3_vysvetlenie: str = (
    "V PyTorch platí to isté ako v JAX: 1D vektor má shape (n,), a `A.T` nič nezmení, "
    "lebo transpozícia je definovaná pre 2D (resp. permutáciu osí) — pri 1D nie je čo prehodiť. "
    "Riešenie je najprv spraviť z vektora maticu (napr. (2,3) alebo (n,1)/(1,n)) a potom transponovať."
)


def uloha_1_4_2() -> torch.Tensor:
    """Analyzuje transpozíciu v PyTorch a opraví chyby.

    Returns:
        torch.Tensor: Transponovaná matica (3, 2) vytvorená z vektora (6,)
    """

    # Pokus 1
    A1 = torch.tensor([[1, 2], [3, 4], [5, 6]])
    result1 = A1.T

    # Pokus 2
    A2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
    result2 = torch.transpose(A2, 0, 1)

    # Pokus 3
    A3 = torch.tensor([1, 2, 3, 4, 5, 6])
    result3 = A3.T

    global uloha_1_4_2_pokus1_shape, uloha_1_4_2_pokus2_shape, uloha_1_4_2_pokus3_shape
    uloha_1_4_2_pokus1_shape = tuple(result1.shape)
    uloha_1_4_2_pokus2_shape = tuple(result2.shape)
    uloha_1_4_2_pokus3_shape = tuple(result3.shape)

    # Oprava: (6,) -> (2,3) -> transpose -> (3,2)
    matica_2_3 = A3.reshape(2, 3)
    transponovana_matica = matica_2_3.T
    return transponovana_matica


uloha_1_4_3_su_rovnake: bool | None = None
uloha_1_4_3_cas_manual: float | None = None
uloha_1_4_3_cas_vectorized: float | None = None
uloha_1_4_3_result3_error: str | None = None

uloha_1_4_3_analyza: str = (
    "V PyTorch broadcasting funguje podobne: (m,n) * (n,) škáluje stĺpce. "
    "`v.reshape(1,n)` je explicitne to isté. `v.reshape(n,1)` s (m,n) zlyhá, lebo sa nedá broadcastovať "
    "(m vs n v ne-zodpovedajúcej osi). Na škálovanie riadkov vektorom `w` tvaru (m,) použite `w.reshape(m,1)` alebo `w[:, None]`."
)


def uloha_1_4_3(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Implementuje broadcasting v PyTorch "ručne" pomocou for loop.

    Args:
        A: Matica (3, 4) (všeobecne (m, n))
        v: Vektor (4,) (všeobecne (n,))

    Returns:
        torch.Tensor: Výsledok broadcasting A * v
    """

    if A.ndim != 2:
        raise ValueError("A musí byť 2D matica")
    if v.ndim != 1:
        raise ValueError("v musí byť 1D vektor")
    if A.shape[1] != v.shape[0]:
        raise ValueError("Počet stĺpcov A musí sedieť s dĺžkou v")

    m, n = A.shape

    start = time.perf_counter()
    result_manual_rows: list[list[torch.Tensor]] = []
    for i in range(m):
        row: list[torch.Tensor] = []
        for j in range(n):
            row.append(A[i, j] * v[j])
        result_manual_rows.append(row)
    result_manual = torch.stack([torch.stack(r) for r in result_manual_rows])
    elapsed_manual = time.perf_counter() - start

    start = time.perf_counter()
    result1 = A * v
    elapsed_vectorized = time.perf_counter() - start

    result2 = A * v.reshape(1, -1)

    err_msg: str | None
    try:
        _bad = A * v.reshape(-1, 1)
        # Force evaluation (primárne pre CUDA; na CPU je to aj tak eager)
        if _bad.is_cuda:
            torch.cuda.synchronize(device=_bad.device)
        err_msg = None
    except Exception as e:  # noqa: BLE001
        err_msg = f"{type(e).__name__}: {e}"

    same = bool(torch.equal(result_manual, result1)) and bool(torch.equal(result1, result2))

    global uloha_1_4_3_su_rovnake, uloha_1_4_3_cas_manual, uloha_1_4_3_cas_vectorized, uloha_1_4_3_result3_error
    uloha_1_4_3_su_rovnake = same
    uloha_1_4_3_cas_manual = elapsed_manual
    uloha_1_4_3_cas_vectorized = elapsed_vectorized
    uloha_1_4_3_result3_error = err_msg

    # Bonus: škálovanie riadkov vektorom w (m,)
    w = torch.arange(1, m + 1, device=A.device, dtype=A.dtype) * 10
    _row_scaled = A * w.reshape(m, 1)

    return result1


if __name__ == "__main__":
    # Rýchla kontrola, že importy a riešenie fungujú.
    out = uloha_1_1()
    print("uloha_1_1() ->", out)
    print("Same?:", uloha_1_1_su_rovnake)

    out2 = uloha_1_2()
    print("uloha_1_2() ->", out2)
    print(
        "shapes:",
        uloha_1_2_pokus1_shape,
        uloha_1_2_pokus2_shape,
        uloha_1_2_pokus3_shape,
    )

    A = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    v = jnp.array([2, 3, 4, 5])
    out3 = uloha_1_3(A, v)
    print("uloha_1_3() ->", out3)
    print("Same?:", uloha_1_3_su_rovnake)
    print("timing manual/vectorized:", uloha_1_3_cas_manual, uloha_1_3_cas_vectorized)
    print("result3 error:", uloha_1_3_result3_error)

    out4_1 = uloha_1_4_1()
    print("uloha_1_4_1() ->", out4_1)
    print("Same?:", uloha_1_4_1_su_rovnake)

    out4_2 = uloha_1_4_2()
    print("uloha_1_4_2() ->", out4_2)
    print(
        "shapes:",
        uloha_1_4_2_pokus1_shape,
        uloha_1_4_2_pokus2_shape,
        uloha_1_4_2_pokus3_shape,
    )

    A_t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    v_t = torch.tensor([2, 3, 4, 5])
    out4_3 = uloha_1_4_3(A_t, v_t)
    print("uloha_1_4_3() ->", out4_3)
    print("Same?:", uloha_1_4_3_su_rovnake)
    print("timing manual/vectorized:", uloha_1_4_3_cas_manual, uloha_1_4_3_cas_vectorized)
    print("result3 error:", uloha_1_4_3_result3_error)

      

