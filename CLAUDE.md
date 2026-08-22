# atlas_ai_v2 (RobotEye) — contexto para Claude Code

A arquitetura interna, orçamento de performance no Raspberry Pi 5 e as
convenções de estilo deste repositório já estão documentadas em detalhe em
**[`.claude/agents/roboteye-keeper.md`](./.claude/agents/roboteye-keeper.md)**
— leia aquilo primeiro para qualquer mudança de código. Este arquivo cobre
só o que falta: a relação (ou falta dela) com o resto do projeto do robô.

## Desacoplado de `orquestrador` — de propósito

Este repositório **não** fala com o barramento MQTT do `orquestrador` (o
"corpo" do robô: motores, GPS, Wi-Fi). Confirmado por leitura de todo
`src/roboteye/`: nenhuma menção a MQTT, porta serial, GPIO, ou aos tópicos
de `robo_common/topics.py`. `core/assistant.py:1` usa a palavra
"orquestrador" só como substantivo comum em português (descreve que a
classe `Assistant` liga LLM + memória + voz), não como referência ao
repositório `orquestrador`.

As duas partes compartilham apenas a marca "Atlas" (mesma persona, ver
`persona/atlas.md`). Se um dia fizer sentido integrá-las — por exemplo, o
RobotEye reagir a `robo/telemetria/bateria` ou falar quando o `orquestrador`
receber `{"tipo":"voz",...}` via `robo/voz/falar` — isso é trabalho novo, não
algo que já existe e só não foi documentado. Veja
`../MAPA-COMUNICACAO.md` para o mapa completo do que existe hoje.

## Já fortemente orientado a objetos

Se for usar este projeto como exemplo de POO para a disciplina do curso,
os pontos mais fortes (com arquivo:linha) são:

- **Herança + polimorfismo clássicos**: `core/events.py::Event` (dataclass
  base, `frozen=True`) com 8 subclasses (`UserMessage`, `ThinkingStarted`,
  `AssistantReply`, `SpeechStarted`, `SpeechFinished`, `ErrorOccurred`,
  `Notice`, `Shutdown`); `EventBus.publish` despacha por `isinstance`. É o
  exemplo mais "de livro-texto" do projeto inteiro.
- **Factory Method**: `llm/factory.py::create_llm_client` e
  `speech/factory.py::create_tts_engine` escolhem a implementação concreta
  (`match backend: case "ollama"/"echo"` e `case "piper"/"kokoro"/"edge"/"null"`).
- **Decorator**: `speech/fallback.py::FallbackEngine` embrulha dois
  `TTSEngine` (primário + reserva) atrás da mesma interface.
- **Abstração via `typing.Protocol`** (não `abc.ABC`): `llm/base.py::LLMClient`
  e `speech/base.py::TTSEngine` — importante notar essa diferença se a
  disciplina exigir herança nominal clássica (`class X(Base):`); Protocol é
  "duck typing" estruturalmente tipado, não herança nominal.
- **Injeção de dependência + encapsulamento**: `core/assistant.py::Assistant`
  recebe todos os colaboradores (`llm`, `memory`, `speaker`, `bus`, `persona`)
  no `__init__`, guarda tudo como atributo privado, e implementa o protocolo
  de context manager (`__enter__`/`__exit__`).
- **Dataclasses `frozen=True, slots=True`** em todo `config.py`, com
  `from_env()` como construtor alternativo (classmethod).
