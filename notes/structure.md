# Thesis Structure - Notes and Ideas

## Preface (?)

Perhaps I should preface with a discussion on the distinction between safety and
security. Currently, nearly all papers discuss jailbreaking in the context of
safety rather than security which at first might seem innocuous, but I think
carries deeper consequences.

We generally associate safety with a lack of risk or physical harm in the sense
that one cannot inflict harm onto themselves or by some accident be harmed.
Security is generally considered to be applicable when determining to what
extent a system is resistant to an intentional, external source of, harm.

Clearly, in the context of jailbreaking, we are actively trying to inflict harm.
Regardless of how we exactly define harm in this context, we are intentionally
inflicting it, since we are aware that what we are doing is not in-line with how
the system is supposed to be used.

Furthermore, I would argue that a person actively trying to jailbreak a system
is _not_ harming themselves. At least, not intentionally. An adversary's aim is
not to inflict harm upon themselves, but to gain information that the system
should or would otherwise not provide. This is often, if not exclusively, done
to gain harmful information. Since these systems are explicitly built to avoid
providing harmful information, I would consider this inflicting a harm. As the
adversary, which is an external entity, is intentionally inflicting harm, I
would consider this a security issue, not a safety one.

One might argue: "Who cares?". Besides me, everyone that somehow is dependent on
the correct functioning of such systems should. Labelling such problems as
safety carries with it the implication that systems are preventing their users
from harming themselves using the system. The problem with this, however, is
that we're not talking about physical safety. No, here, we consider some
non-physical, say, intellectual safety.

While the extent to which something is physically safe is debatable, it's
relatively non-disputed when something inflicts physical harm; when it hurts
you. Intellectual harm, however, is very different. Rarely ever do we talk about
the direct effects, but about the broader impact of such harm.  Expressing what
constitutes as harm in an intellectual or information context means expressing a
certain philosophy on ethics. Considering the safety of an LLM is therefore
considering the ethics.

Zooming out again to jailbreaking and its relation to AI safety, I would say
that whoever concludes that a machine learning algorithm is unsafe because it
fails to prevent a jailbreaking attack, is implicitly agreeing with the ethics
of the creators of the system. This is not a problem per se, but the issue, in
my opinion lies that this is done pervasively unconsciously. We're implicitly
agreeing on ethics we might not even agree with!

<!--  Rant about agreeing with companies, or a general lack of critical thinking?  -->

This is not to say that anyone has the intention to do so, and the discussion
on safety is an incredibly important one. But the issue here lies in the wrong
type of discussion. If we aim to determine whether a system is aligned with
its requirements, we should consider it a security issue. If we all actually
put as much thought into the discussion of AI safety instead of that of AI
security, our understanding of these systems as well as ourselves might
progress a whole lot faster.
