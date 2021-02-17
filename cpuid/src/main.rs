use cpuid::MicroArchitecture;

fn main() {
    println!("{:?}", MicroArchitecture::get_micro_architecture());
}
