simple Txc1first
{
gates:
input in;
output out;
}
network Tictocfirst
{
@display("bgb=390,315");
submodules:
tic: Txc1first {
@display("p=59,221");
}
toc: Txc1first {
@display("p=310,61");
}
connections:
tic.out --> { delay = 100ms; } --> toc.in;
tic.in <-- { delay = 100ms; } <-- toc.out;
}