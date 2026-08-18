"""Testes da animação dos olhos (lógica pura, sem pygame)."""

from __future__ import annotations

import random
from itertools import pairwise

import pytest

from roboteye.face.animator import EyeAnimator, EyeFrame
from roboteye.face.expressions import Expression
from roboteye.face.layout import EyeLayout

PASSO = 1 / 60


def avancar(animator: EyeAnimator, segundos: float) -> EyeFrame:
    """Roda a animação por um tempo e devolve o último quadro."""
    frame = animator.update(PASSO)
    decorrido = PASSO
    while decorrido < segundos:
        frame = animator.update(PASSO)
        decorrido += PASSO
    return frame


def coletar(animator: EyeAnimator, segundos: float) -> list[EyeFrame]:
    """Todos os quadros de um intervalo."""
    return [animator.update(PASSO) for _ in range(int(segundos * 60))]


@pytest.fixture
def animator() -> EyeAnimator:
    return EyeAnimator(idle_animations=False, rng=random.Random(42))


class TestPiscada:
    def test_olhos_comecam_abertos(self, animator: EyeAnimator) -> None:
        frame = animator.update(PASSO)
        assert frame.left.openness > 0.9

    def test_piscada_fecha_os_olhos(self, animator: EyeAnimator) -> None:
        animator.blink_now()
        alturas = [f.left.openness for f in coletar(animator, 0.15)]
        assert min(alturas) < 0.1

    def test_olhos_reabrem_depois(self, animator: EyeAnimator) -> None:
        animator.blink_now()
        frame = avancar(animator, 0.5)
        assert frame.left.openness > 0.9

    def test_fecha_mais_rapido_do_que_abre(self, animator: EyeAnimator) -> None:
        """Assimetria proposital: é assim que uma piscada de verdade acontece."""
        animator.blink_now()
        alturas = [f.left.openness for f in coletar(animator, 0.4)]

        fundo = alturas.index(min(alturas))
        reaberto = next(i for i, h in enumerate(alturas[fundo:], fundo) if h > 0.95)

        assert fundo < (reaberto - fundo), "abrir deveria levar mais tempo que fechar"

    def test_os_dois_olhos_piscam_juntos(self, animator: EyeAnimator) -> None:
        animator.blink_now()
        for frame in coletar(animator, 0.3):
            assert frame.left.openness == pytest.approx(frame.right.openness, abs=0.02)

    def test_a_reabertura_e_suave(self, animator: EyeAnimator) -> None:
        """O que estraga uma animação não é a velocidade, é a mudança brusca dela.

        A pálpebra pode percorrer bastante altura num quadro — é uma piscada,
        deve ser rápida. O que não pode é a velocidade saltar de um quadro para
        o outro. A reabertura é a fase longa e visível, onde isso apareceria.
        """
        animator.blink_now()
        alturas = [f.left.openness for f in coletar(animator, 0.6)]

        fundo = alturas.index(min(alturas))
        reabertura = alturas[fundo:]

        velocidades = [b - a for a, b in pairwise(reabertura)]
        aceleracoes = [abs(b - a) for a, b in pairwise(velocidades)]

        # 0,08 passa folgado numa curva que arranca do repouso e aperta uma que
        # comece na velocidade maxima (essa chegava a 0,16).
        assert max(aceleracoes) < 0.08

    def test_fica_um_instante_fechada(self, animator: EyeAnimator) -> None:
        """A pausa no fundo evita que a pálpebra inverta o sentido num quadro só."""
        animator.blink_now()
        alturas = [f.left.openness for f in coletar(animator, 0.6)]
        assert sum(1 for h in alturas if h < 0.08) >= 2


class TestPiscadaComExpressao:
    """A piscada tem de funcionar a partir de qualquer expressão, não só do neutro."""

    @pytest.mark.parametrize(
        "expressao",
        [Expression.NEUTRAL, Expression.HAPPY, Expression.ANGRY, Expression.TIRED],
    )
    def test_fecha_a_partir_de_qualquer_expressao(
        self, animator: EyeAnimator, expressao: Expression
    ) -> None:
        animator.set_mood(expressao)
        avancar(animator, 0.6)  # deixa o morph terminar

        animator.blink_now()
        assert min(f.left.openness for f in coletar(animator, 0.3)) < 0.1

    @pytest.mark.parametrize("expressao", [Expression.HAPPY, Expression.LAUGH])
    def test_a_palpebra_de_baixo_nunca_recua(
        self, animator: EyeAnimator, expressao: Expression
    ) -> None:
        """O sorriso mora na pálpebra de baixo; piscar não pode desmanchá-lo.

        Era o defeito: com alvo fixo, a pálpebra inferior *descia* de 0,44 para
        0,16 durante a piscada, e o sorriso se refazia depois. Ao vivo, isso
        aparecia como a expressão piscando junto com o olho.
        """
        animator.set_mood(expressao)
        repouso = avancar(animator, 0.6).left.bottom_lid

        animator.blink_now()
        for frame in coletar(animator, 0.5):
            assert frame.left.bottom_lid >= repouso - 0.01

    def test_a_fresta_final_independe_da_expressao(self, animator: EyeAnimator) -> None:
        """O traço que sobra no fundo tem a mesma espessura em qualquer expressão."""

        def fundo(expressao: Expression) -> float:
            face = EyeAnimator(idle_animations=False, rng=random.Random(42))
            face.set_mood(expressao)
            avancar(face, 0.6)
            face.blink_now()
            return min(f.left.openness for f in coletar(face, 0.5))

        neutro = fundo(Expression.NEUTRAL)
        for expressao in (Expression.HAPPY, Expression.ANGRY, Expression.TIRED):
            assert fundo(expressao) == pytest.approx(neutro, abs=0.02)

    @pytest.mark.parametrize("expressao", [Expression.ANGRY, Expression.TIRED])
    def test_a_inclinacao_afrouxa_ate_fechar(
        self, animator: EyeAnimator, expressao: Expression
    ) -> None:
        """Olho fechado não tem canto caído — senão a piscada vira um corte em diagonal."""
        animator.set_mood(expressao)
        repouso = avancar(animator, 0.6).left
        assert abs(repouso.top_lid_slant) > 0.5  # a expressão realmente inclina

        animator.blink_now()
        quadros = coletar(animator, 0.5)
        fundo = min(quadros, key=lambda f: f.left.openness)
        assert abs(fundo.left.top_lid_slant) < abs(repouso.top_lid_slant) * 0.2


class TestSono:
    def test_dormir_fecha_os_olhos(self, animator: EyeAnimator) -> None:
        animator.sleep()
        frame = avancar(animator, 1.0)

        assert frame.left.openness < 0.1
        assert frame.expression is Expression.SLEEP

    def test_acordar_reabre(self, animator: EyeAnimator) -> None:
        animator.sleep()
        avancar(animator, 1.0)
        animator.wake()
        frame = avancar(animator, 2.0)

        assert not animator.is_sleeping
        assert frame.left.openness > 0.8

    def test_toggle_alterna(self, animator: EyeAnimator) -> None:
        animator.toggle_sleep()
        assert animator.is_sleeping
        animator.toggle_sleep()
        assert not animator.is_sleeping

    def test_dormir_centraliza_o_olhar(self, animator: EyeAnimator) -> None:
        animator.sleep()
        frame = avancar(animator, 1.0)
        assert abs(frame.left.offset_x) < 20

    def test_dormindo_nao_pisca(self, animator: EyeAnimator) -> None:
        """Pálpebra descendo sobre um olho já fechado não lê como piscar, lê como defeito."""
        animator.sleep()
        avancar(animator, 1.0)

        repouso = animator.update(PASSO).left.openness
        animator.blink_now()

        for frame in coletar(animator, 0.5):
            assert frame.left.openness == pytest.approx(repouso, abs=0.01)

    def test_dormindo_nao_pisca_sozinha(self, animator: EyeAnimator) -> None:
        animator.sleep()
        avancar(animator, 1.0)

        repouso = animator.update(PASSO).left.openness
        for frame in coletar(animator, 20.0):  # muito além do intervalo de piscada
            assert frame.left.openness == pytest.approx(repouso, abs=0.01)

    def test_piscada_dobrada_nao_sobrevive_ao_sono(self, animator: EyeAnimator) -> None:
        """A fila de piscadas morre ao dormir, em vez de disparar já com ela dormindo."""
        animator.blink_now()
        animator._queued_blinks = 1  # como se o sorteio tivesse pedido a dobrada
        animator.sleep()

        avancar(animator, 1.0)
        repouso = animator.update(PASSO).left.openness
        for frame in coletar(animator, 2.0):
            assert frame.left.openness == pytest.approx(repouso, abs=0.01)


class TestAtividades:
    def test_pensar_olha_para_cima(self, animator: EyeAnimator) -> None:
        animator.set_activity(Expression.THINKING)
        frame = avancar(animator, 1.0)

        assert frame.left.offset_y < -40
        assert frame.expression is Expression.THINKING

    def test_pensar_comeca_com_antecipacao(self, animator: EyeAnimator) -> None:
        """O olhar dá uma caída antes de subir — princípio de antecipação."""
        animator.set_activity(Expression.THINKING)
        alturas = [f.left.offset_y for f in coletar(animator, 0.2)]
        assert max(alturas) > 5, "faltou a caída inicial"

    def test_pensar_deixa_os_olhos_diferentes(self, animator: EyeAnimator) -> None:
        """A assimetria é o que transforma a forma em intenção."""
        animator.set_activity(Expression.THINKING)
        frame = avancar(animator, 1.0)
        assert abs(frame.left.top_lid - frame.right.top_lid) > 0.05

    def test_pensar_move_o_olhar_de_um_lado_a_outro(self, animator: EyeAnimator) -> None:
        animator.set_activity(Expression.THINKING)
        posicoes = [f.left.offset_x for f in coletar(animator, 5.0)]
        assert max(posicoes) > 50 and min(posicoes) < -50

    def test_falar_faz_os_olhos_pulsarem(self, animator: EyeAnimator) -> None:
        animator.set_activity(Expression.SPEAKING)
        alturas = [f.left.height for f in coletar(animator, 2.0)]
        assert max(alturas) - min(alturas) > 0.03

    def test_pulso_da_fala_e_discreto(self, animator: EyeAnimator) -> None:
        """Acima de ~15% de variação a face treme em vez de respirar."""
        animator.set_activity(Expression.SPEAKING)
        alturas = [f.left.height for f in coletar(animator, 3.0)]
        assert max(alturas) - min(alturas) < 0.20

    def test_falar_estica_e_achata(self, animator: EyeAnimator) -> None:
        """Quando o olho achata, ele alarga: squash and stretch."""
        animator.set_activity(Expression.SPEAKING)
        quadros = coletar(animator, 2.0)[30:]  # ignora o "pop" de entrada

        mais_alto = max(quadros, key=lambda f: f.left.height)
        mais_baixo = min(quadros, key=lambda f: f.left.height)

        assert mais_alto.left.width < mais_baixo.left.width

    def test_o_olho_segue_a_amplitude_do_audio(self, animator: EyeAnimator) -> None:
        """Falar alto abre mais o olho do que falar baixo.

        E a diferenca entre uma face que se move *enquanto* fala e uma que se
        move *junto com* a fala: sem isso o pulso e sempre o mesmo, tanto numa
        palavra longa quanto numa pausa.
        """
        animator.set_activity(Expression.SPEAKING)

        animator.set_speech_level(0.05)
        baixo = [f.left.height for f in coletar(animator, 1.0)][-10:]

        animator.set_speech_level(1.0)
        alto = [f.left.height for f in coletar(animator, 1.0)][-10:]

        assert max(alto) > max(baixo)

    def test_silencio_no_meio_da_fala_para_o_olho(self, animator: EyeAnimator) -> None:
        """Numa pausa entre frases o olho descansa, em vez de seguir pulsando."""
        animator.set_activity(Expression.SPEAKING)
        animator.set_speech_level(0.0)
        coletar(animator, 1.0)  # deixa o suavizador assentar

        alturas = [f.left.height for f in coletar(animator, 1.0)]
        assert max(alturas) - min(alturas) < 0.01

    def test_sem_medicao_o_olho_volta_a_pulsar_sozinho(self, animator: EyeAnimator) -> None:
        """Um motor que nao produz PCM nao pode deixar a face imovel.

        `None` (ninguem esta medindo) tem de ser tratado diferente de `0.0`
        (esta em silencio) — senao a face congela ao falar por um caminho de
        audio que nao informa amplitude.
        """
        animator.set_activity(Expression.SPEAKING)
        animator.set_speech_level(None)

        alturas = [f.left.height for f in coletar(animator, 2.0)]
        assert max(alturas) - min(alturas) > 0.03

    def test_atividade_tem_prioridade_sobre_o_humor(self, animator: EyeAnimator) -> None:
        animator.set_mood(Expression.HAPPY)
        animator.set_activity(Expression.SPEAKING)
        assert animator.current_expression is Expression.SPEAKING

    def test_humor_volta_ao_fim_da_atividade(self, animator: EyeAnimator) -> None:
        animator.set_mood(Expression.HAPPY)
        animator.set_activity(Expression.SPEAKING)
        animator.set_activity(None)
        assert animator.current_expression is Expression.HAPPY

    def test_atividade_acorda(self, animator: EyeAnimator) -> None:
        animator.sleep()
        animator.set_activity(Expression.THINKING)
        assert not animator.is_sleeping

    def test_atividade_invalida_e_rejeitada(self, animator: EyeAnimator) -> None:
        with pytest.raises(ValueError, match="atividade"):
            animator.set_activity(Expression.HAPPY)


class TestHumores:
    def test_riso_vira_alegria(self, animator: EyeAnimator) -> None:
        animator.set_mood(Expression.LAUGH)
        avancar(animator, 2.5)
        assert animator.current_expression is Expression.HAPPY

    def test_tontura_volta_ao_neutro(self, animator: EyeAnimator) -> None:
        animator.set_mood(Expression.DIZZY)
        avancar(animator, 3.2)
        assert animator.current_expression is Expression.NEUTRAL

    def test_riso_sacode_na_vertical(self, animator: EyeAnimator) -> None:
        animator.set_mood(Expression.LAUGH)
        alturas = [f.left.offset_y for f in coletar(animator, 1.0)]
        assert max(alturas) - min(alturas) > 20

    def test_tontura_sacode_os_olhos_fora_de_fase(self, animator: EyeAnimator) -> None:
        animator.set_mood(Expression.DIZZY)
        quadros = coletar(animator, 1.0)
        assert any(abs(f.left.offset_x - f.right.offset_x) > 8 for f in quadros)

    def test_expressoes_de_atividade_nao_viram_humor(self, animator: EyeAnimator) -> None:
        animator.set_mood(Expression.THINKING)
        assert animator.current_expression is Expression.NEUTRAL

    def test_troca_de_humor_e_gradual(self, animator: EyeAnimator) -> None:
        """O defeito antigo: expressão mudava de um quadro para o outro."""
        avancar(animator, 0.5)
        animator.set_mood(Expression.ANGRY)
        quadros = coletar(animator, 0.6)

        saltos = [abs(b.left.top_lid - a.left.top_lid) for a, b in pairwise(quadros)]
        assert max(saltos) < 0.05, "a pálpebra deveria descer aos poucos"
        assert quadros[-1].left.top_lid > 0.2, "e chegar ao destino"


class TestVida:
    def test_o_olhar_nunca_fica_completamente_parado(self, animator: EyeAnimator) -> None:
        """Microssacadas: olho de verdade treme mesmo fixando um ponto."""
        posicoes = {round(f.left.offset_x, 3) for f in coletar(animator, 6.0)}
        assert len(posicoes) > 12

    def test_a_altura_nunca_fica_travada(self, animator: EyeAnimator) -> None:
        """Respiração: a altura oscila de leve o tempo todo, mesmo em repouso."""
        alturas = [f.left.openness for f in coletar(animator, 6.0)]
        assert len({round(h, 4) for h in alturas}) > 50

    def test_a_altura_em_repouso_fica_perto_do_natural(self, animator: EyeAnimator) -> None:
        """Respiração e curiosidade somadas não podem inchar o olho."""
        em_repouso = [h for h in (f.left.openness for f in coletar(animator, 8.0)) if h > 0.85]
        assert max(em_repouso) < 1.25

    def test_o_olho_direito_atrasa_em_relacao_ao_esquerdo(self, animator: EyeAnimator) -> None:
        quadros = coletar(animator, 8.0)
        assert any(abs(f.left.offset_x - f.right.offset_x) > 5 for f in quadros)

    def test_pisca_sozinha_de_tempos_em_tempos(self, animator: EyeAnimator) -> None:
        alturas = [f.left.openness for f in coletar(animator, 20.0)]
        assert min(alturas) < 0.2

    def test_troca_de_humor_sozinha_quando_ociosa(self) -> None:
        animator = EyeAnimator(idle_animations=True, rng=random.Random(1))
        vistos = {animator.update(PASSO).expression for _ in range(60 * 90)}
        assert len(vistos) > 1

    def test_nao_troca_de_humor_com_as_animacoes_desligadas(self, animator: EyeAnimator) -> None:
        vistos = {animator.update(PASSO).expression for _ in range(60 * 60)}
        assert vistos == {Expression.NEUTRAL}


class TestEstabilidade:
    def test_dt_grande_nao_teleporta(self, animator: EyeAnimator) -> None:
        frame = animator.update(5.0)
        assert 0.0 <= frame.left.height <= 2.0

    def test_parametros_ficam_em_faixas_sensatas(self, animator: EyeAnimator) -> None:
        animator.set_activity(Expression.SPEAKING)
        for frame in coletar(animator, 10.0):
            for shape in (frame.left, frame.right):
                assert 0.0 <= shape.height <= 1.6
                assert 0.5 <= shape.width <= 1.6
                assert 0.0 <= shape.top_lid <= 1.0
                assert 0.0 <= shape.bottom_lid <= 1.0
                assert abs(shape.offset_x) < 600
                assert abs(shape.offset_y) < 300

    def test_sequencia_longa_de_comandos_nao_quebra(self, animator: EyeAnimator) -> None:
        rng = random.Random(99)
        acoes = [
            lambda: animator.set_mood(rng.choice([e for e in Expression if e.is_mood])),
            lambda: animator.set_activity(
                rng.choice([Expression.THINKING, Expression.SPEAKING, None])
            ),
            animator.blink_now,
            animator.toggle_sleep,
        ]
        for _ in range(300):
            rng.choice(acoes)()
            for _ in range(6):
                animator.update(PASSO)


class TestLayout:
    def test_escala_proporcional(self) -> None:
        layout = EyeLayout.for_screen(2560, 1440)
        assert layout.scale == pytest.approx(1.0)
        assert layout.eye_width == 640

    def test_tela_pequena_encolhe_tudo(self) -> None:
        layout = EyeLayout.for_screen(800, 480)
        assert layout.eye_width < 640

    def test_olhos_ficam_dentro_da_tela(self) -> None:
        layout = EyeLayout.for_screen(1280, 720)
        assert 0 < layout.left_eye_x < layout.right_eye_x < 1280
        assert 0 < layout.eye_center_y < 720
